import numpy as np
import torch

from normalizing_flow import RealNVPFlow


def empirical_cdf_score(value, sorted_calibration_values):
    calibration = np.asarray(sorted_calibration_values, dtype=np.float64).reshape(-1)
    if calibration.size == 0:
        raise ValueError("calibration values must be non-empty")
    if not np.all(np.isfinite(calibration)):
        raise ValueError("calibration values must be finite")
    return float(np.searchsorted(calibration, float(value), side="right") / calibration.size)


class NormalizingFlowActorSwitcher(object):
    """OOD-style switcher using nominal-state likelihood from a frozen RealNVP."""

    def __init__(
        self,
        standard_policy,
        interaction_policy,
        flow_checkpoint,
        device,
    ):
        self.standard_policy = standard_policy
        self.interaction_policy = interaction_policy
        self.device = torch.device(device)

        payload = torch.load(flow_checkpoint, map_location=self.device, weights_only=False)
        model_config = dict(payload.get("model_config", {}))
        self.flow = RealNVPFlow(**model_config).to(self.device)
        self.flow.load_state_dict(payload["model_state_dict"])
        self.flow.eval()

        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.threshold_nll = float(payload["threshold_nll"])
        self.calibration_nll_sorted = np.asarray(
            payload["calibration_nll_sorted"], dtype=np.float64
        )
        self.input_dim = int(model_config.get("input_dim", len(self.feature_mean)))
        if self.feature_mean.shape != (self.input_dim,) or self.feature_std.shape != (
            self.input_dim,
        ):
            raise ValueError("normalizing-flow statistics do not match input dimension")
        if (
            not np.all(np.isfinite(self.feature_mean))
            or not np.all(np.isfinite(self.feature_std))
            or np.any(self.feature_std <= 0.0)
            or not np.isfinite(self.threshold_nll)
        ):
            raise ValueError("normalizing-flow checkpoint contains invalid statistics")
        if self.calibration_nll_sorted.ndim != 1 or self.calibration_nll_sorted.size == 0:
            raise ValueError("normalizing-flow checkpoint lacks calibration NLL values")

        self.current_mode = {}
        self.switch_count = 0
        self.score_sum = 0.0
        self.score_count = 0

    def reset(self, agent_names):
        self.current_mode = {name: "standard" for name in agent_names}
        self.switch_count = 0
        self.score_sum = 0.0
        self.score_count = 0

    @torch.no_grad()
    def _score_state(self, state):
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.shape != (self.input_dim,):
            raise ValueError("normalizing-flow switch expects a 24-D Actor state")
        if not np.all(np.isfinite(values)):
            raise ValueError("normalizing-flow switch received non-finite state")
        normalized = ((values - self.feature_mean) / self.feature_std).astype(
            np.float32
        )
        tensor = torch.from_numpy(normalized.reshape(1, -1)).to(self.device)
        nll = float(self.flow.negative_log_likelihood(tensor).cpu().item())
        if not np.isfinite(nll):
            raise ValueError("normalizing-flow switch produced non-finite NLL")
        return nll, empirical_cdf_score(nll, self.calibration_nll_sorted)

    def choose_action(self, env, name, state, logical_time=None):
        del env, logical_time
        if name not in self.current_mode:
            raise ValueError("normalizing-flow switch must be reset before use")
        nll, score = self._score_state(state)
        mode = "dense" if nll > self.threshold_nll else "standard"
        previous = self.current_mode.get(name, "standard")
        if mode != previous:
            self.switch_count += 1
        self.current_mode[name] = mode
        policy = self.interaction_policy if mode == "dense" else self.standard_policy
        self.score_sum += score
        self.score_count += 1
        return policy.get_action(np.asarray(state)), mode, score, {"nll": nll}

    def episode_stats(self):
        return {
            "switches": int(self.switch_count),
            "mean_probability": (
                self.score_sum / self.score_count if self.score_count else 0.0
            ),
        }
