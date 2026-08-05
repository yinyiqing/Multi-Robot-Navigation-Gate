import numpy as np


class MinLidarActorSwitcher(object):
    """Deployable rule baseline that treats every nearby LiDAR return alike."""

    def __init__(
        self,
        standard_policy,
        interaction_policy,
        lidar_bins=20,
        switch_on_distance=2.0,
        switch_off_distance=2.2,
        minimum_hold_steps=3,
    ):
        if int(lidar_bins) < 1:
            raise ValueError("lidar_bins must be positive")
        if float(switch_on_distance) <= 0.0:
            raise ValueError("switch_on_distance must be positive")
        if float(switch_off_distance) <= float(switch_on_distance):
            raise ValueError("switch_off_distance must exceed switch_on_distance")
        if int(minimum_hold_steps) < 0:
            raise ValueError("minimum_hold_steps cannot be negative")
        self.standard_policy = standard_policy
        self.interaction_policy = interaction_policy
        self.lidar_bins = int(lidar_bins)
        self.switch_on_distance = float(switch_on_distance)
        self.switch_off_distance = float(switch_off_distance)
        self.minimum_hold_steps = int(minimum_hold_steps)
        self.current_mode = {}
        self.mode_steps = {}
        self.switch_count = 0

    def reset(self, agent_names):
        self.current_mode = {name: "standard" for name in agent_names}
        self.mode_steps = {name: 0 for name in agent_names}
        self.switch_count = 0

    def choose_action(self, env, name, state):
        del env
        values = np.asarray(state, dtype=float).reshape(-1)
        if len(values) < self.lidar_bins:
            raise ValueError("actor state does not contain all LiDAR bins")
        lidar = values[: self.lidar_bins]
        if not np.all(np.isfinite(lidar)):
            raise ValueError("actor state contains non-finite LiDAR values")
        minimum_distance = float(np.min(lidar))
        mode = self.current_mode.get(name, "standard")
        mode_steps = self.mode_steps.get(name, 0)

        next_mode = mode
        if mode == "standard" and minimum_distance <= self.switch_on_distance:
            next_mode = "dense"
        elif (
            mode == "dense"
            and mode_steps >= self.minimum_hold_steps
            and minimum_distance >= self.switch_off_distance
        ):
            next_mode = "standard"

        if next_mode != mode:
            self.switch_count += 1
            mode_steps = 0
        self.current_mode[name] = next_mode
        self.mode_steps[name] = mode_steps + 1
        policy = (
            self.interaction_policy if next_mode == "dense" else self.standard_policy
        )
        return policy.get_action(values), next_mode, minimum_distance, 1

    def episode_stats(self):
        return {"switches": int(self.switch_count), "mean_probability": 0.0}
