import numpy as np


class ConflictPairYieldOracle:
    def __init__(
        self,
        base_policy,
        stop_distance=1.2,
        release_distance=1.4,
        max_yield_steps=20,
    ):
        if stop_distance <= 0.0:
            raise ValueError("stop_distance must be positive")
        if release_distance <= stop_distance:
            raise ValueError("release_distance must exceed stop_distance")
        if max_yield_steps < 1:
            raise ValueError("max_yield_steps must be positive")
        self.base_policy = base_policy
        self.stop_distance = float(stop_distance)
        self.release_distance = float(release_distance)
        self.max_yield_steps = int(max_yield_steps)
        self.reset()

    def reset(self, agent_names=None):
        self.has_yielded = False
        self.released = False
        self.yield_steps = 0

    @staticmethod
    def _conflict_pair(env):
        case = getattr(env, "current_curriculum_case", None)
        if not isinstance(case, dict):
            raise ValueError("Conflict-pair oracle requires a manifest scenario")
        edges = case.get("metrics", {}).get("conflict_edges", [])
        if len(edges) != 1:
            raise ValueError("Conflict-pair oracle requires exactly one conflict edge")
        pair = sorted(str(name) for name in edges[0].get("agents", []))
        if len(pair) != 2:
            raise ValueError("Conflict edge must contain exactly two agents")
        return pair

    def choose_action(self, env, name, state, active_names):
        action = self.base_policy.get_action(np.asarray(state))
        passer, yielder = self._conflict_pair(env)
        if name != yielder or self.released:
            return action, False
        if passer not in active_names:
            self.released = True
            return action, False

        distance = float(
            np.linalg.norm(env.robot_positions[passer] - env.robot_positions[yielder])
        )
        if not self.has_yielded and distance <= self.stop_distance:
            self.has_yielded = True
        if not self.has_yielded:
            return action, False
        if distance >= self.release_distance or self.yield_steps >= self.max_yield_steps:
            self.released = True
            return action, False

        self.yield_steps += 1
        return np.array([-1.0, 0.0], dtype=np.float32), True
