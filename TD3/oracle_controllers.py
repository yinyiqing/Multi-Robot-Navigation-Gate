import math

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


class RightHandPassOracle:
    """Diagnostic controller for a shared right-hand passing convention."""

    def __init__(
        self,
        base_policy,
        activation_distance=1.5,
        release_distance=1.8,
        frontal_angle_degrees=35.0,
        opposing_angle_degrees=150.0,
        min_closing_speed=0.2,
        max_ttc=3.0,
        turn_action=-0.6,
        linear_speed_cap=0.45,
        max_override_steps=20,
    ):
        if activation_distance <= 0.0:
            raise ValueError("activation_distance must be positive")
        if release_distance <= activation_distance:
            raise ValueError("release_distance must exceed activation_distance")
        if not 0.0 < frontal_angle_degrees < 90.0:
            raise ValueError("frontal_angle_degrees must be in (0, 90)")
        if not 90.0 < opposing_angle_degrees <= 180.0:
            raise ValueError("opposing_angle_degrees must be in (90, 180]")
        if min_closing_speed < 0.0:
            raise ValueError("min_closing_speed must be non-negative")
        if max_ttc <= 0.0:
            raise ValueError("max_ttc must be positive")
        if not -1.0 <= turn_action < 0.0:
            raise ValueError("turn_action must be in [-1, 0)")
        if not 0.0 <= linear_speed_cap <= 1.0:
            raise ValueError("linear_speed_cap must be in [0, 1]")
        if max_override_steps < 1:
            raise ValueError("max_override_steps must be positive")

        self.base_policy = base_policy
        self.activation_distance = float(activation_distance)
        self.release_distance = float(release_distance)
        self.frontal_angle = math.radians(float(frontal_angle_degrees))
        self.opposing_angle = math.radians(float(opposing_angle_degrees))
        self.min_closing_speed = float(min_closing_speed)
        self.max_ttc = float(max_ttc)
        self.turn_action = float(turn_action)
        self.linear_speed_cap = float(linear_speed_cap)
        self.max_override_steps = int(max_override_steps)
        self.reset()

    def reset(self, agent_names=None):
        names = agent_names or []
        self.active_neighbor = {str(name): None for name in names}
        self.cooldown_neighbor = {str(name): None for name in names}
        self.override_steps = {str(name): 0 for name in names}

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _is_head_on(self, env, name, other_name):
        offset = env.robot_positions[other_name] - env.robot_positions[name]
        distance = float(np.linalg.norm(offset))
        if distance <= 0.0 or distance > self.activation_distance:
            return False

        ego_yaw = float(env._get_robot_yaw(name))
        other_yaw = float(env._get_robot_yaw(other_name))
        bearing = self._wrap_angle(math.atan2(offset[1], offset[0]) - ego_yaw)
        heading_difference = abs(self._wrap_angle(other_yaw - ego_yaw))
        if (
            abs(bearing) > self.frontal_angle
            or heading_difference < self.opposing_angle
        ):
            return False

        ego_speed = self._forward_speed(env, name)
        other_speed = self._forward_speed(env, other_name)
        ego_velocity = ego_speed * np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
        other_velocity = other_speed * np.array(
            [math.cos(other_yaw), math.sin(other_yaw)]
        )
        closing_speed = -float(
            np.dot(other_velocity - ego_velocity, offset / distance)
        )
        if closing_speed < self.min_closing_speed:
            return False
        return distance / max(closing_speed, 1e-6) <= self.max_ttc

    @staticmethod
    def _forward_speed(env, name):
        odom = env.last_odom.get(name)
        if odom is None:
            return 0.0
        return max(float(odom.twist.twist.linear.x), 0.0)

    def _select_head_on_neighbor(self, env, name, active_names):
        candidates = [
            other_name
            for other_name in active_names
            if other_name != name and self._is_head_on(env, name, other_name)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda other_name: float(
                np.linalg.norm(
                    env.robot_positions[other_name] - env.robot_positions[name]
                )
            ),
        )

    def _should_release(self, env, name, other_name, active_names):
        if other_name not in active_names:
            return True
        offset = env.robot_positions[other_name] - env.robot_positions[name]
        distance = float(np.linalg.norm(offset))
        if distance >= self.release_distance:
            return True
        ego_yaw = float(env._get_robot_yaw(name))
        forward_offset = (
            math.cos(ego_yaw) * float(offset[0])
            + math.sin(ego_yaw) * float(offset[1])
        )
        return forward_offset <= 0.0

    def choose_action(self, env, name, state, active_names):
        action = np.asarray(
            self.base_policy.get_action(np.asarray(state)), dtype=np.float32
        ).copy()
        name = str(name)
        self.active_neighbor.setdefault(name, None)
        self.cooldown_neighbor.setdefault(name, None)
        self.override_steps.setdefault(name, 0)

        cooldown_name = self.cooldown_neighbor[name]
        if cooldown_name is not None:
            if self._should_release(env, name, cooldown_name, active_names):
                self.cooldown_neighbor[name] = None
            else:
                return action, False

        other_name = self.active_neighbor[name]
        if other_name is not None:
            if self._should_release(env, name, other_name, active_names):
                self.active_neighbor[name] = None
                self.override_steps[name] = 0
                other_name = None
            elif self.override_steps[name] >= self.max_override_steps:
                self.active_neighbor[name] = None
                self.cooldown_neighbor[name] = other_name
                self.override_steps[name] = 0
                return action, False

        if other_name is None:
            other_name = self._select_head_on_neighbor(env, name, active_names)
            if other_name is None:
                return action, False
            self.active_neighbor[name] = other_name

        # Actor linear actions use [-1, 1], mapped to physical [0, 1] later.
        raw_linear_cap = 2.0 * self.linear_speed_cap - 1.0
        action[0] = min(float(action[0]), raw_linear_cap)
        action[1] = min(float(action[1]), self.turn_action)
        self.override_steps[name] += 1
        return action, True
