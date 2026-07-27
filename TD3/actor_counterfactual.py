LABEL_AMBIGUOUS = -1
LABEL_GENERALIST = 0
LABEL_STRONG = 1


REQUIRED_OUTCOME_KEYS = {
    "ego_collision",
    "collision_count",
    "ego_target",
    "minimum_ego_clearance",
    "ego_progress",
}


def _validate_outcome(outcome):
    missing = REQUIRED_OUTCOME_KEYS - set(outcome)
    if missing:
        raise ValueError("counterfactual outcome is missing %s" % sorted(missing))
    if int(outcome["collision_count"]) < 0:
        raise ValueError("collision_count must be non-negative")
    for key in ("minimum_ego_clearance", "ego_progress"):
        value = float(outcome[key])
        if value != value or abs(value) == float("inf"):
            raise ValueError("%s must be finite" % key)


def choose_actor_label(
    generalist,
    strong,
    clearance_margin=0.10,
    progress_margin=0.05,
    maximum_progress_regression=0.05,
):
    """Choose an actor from paired short rollouts using fixed priorities."""
    _validate_outcome(generalist)
    _validate_outcome(strong)
    if clearance_margin <= 0.0 or progress_margin <= 0.0:
        raise ValueError("counterfactual margins must be positive")
    if maximum_progress_regression < 0.0:
        raise ValueError("maximum_progress_regression must be non-negative")

    generalist_collision = bool(generalist["ego_collision"])
    strong_collision = bool(strong["ego_collision"])
    if generalist_collision != strong_collision:
        if strong_collision:
            return LABEL_GENERALIST, "ego_collision"
        return LABEL_STRONG, "ego_collision"

    generalist_collisions = int(generalist["collision_count"])
    strong_collisions = int(strong["collision_count"])
    if generalist_collisions != strong_collisions:
        if strong_collisions < generalist_collisions:
            return LABEL_STRONG, "collision_count"
        return LABEL_GENERALIST, "collision_count"

    generalist_target = bool(generalist["ego_target"])
    strong_target = bool(strong["ego_target"])
    if generalist_target != strong_target:
        if strong_target:
            return LABEL_STRONG, "ego_target"
        return LABEL_GENERALIST, "ego_target"

    generalist_clearance = float(generalist["minimum_ego_clearance"])
    strong_clearance = float(strong["minimum_ego_clearance"])
    generalist_progress = float(generalist["ego_progress"])
    strong_progress = float(strong["ego_progress"])
    clearance_delta = strong_clearance - generalist_clearance
    progress_delta = strong_progress - generalist_progress

    if (
        clearance_delta >= clearance_margin
        and progress_delta >= -maximum_progress_regression
    ):
        return LABEL_STRONG, "clearance"
    if (
        clearance_delta <= -clearance_margin
        and progress_delta <= maximum_progress_regression
    ):
        return LABEL_GENERALIST, "clearance"
    if progress_delta >= progress_margin and clearance_delta >= -clearance_margin:
        return LABEL_STRONG, "progress"
    if progress_delta <= -progress_margin and clearance_delta <= clearance_margin:
        return LABEL_GENERALIST, "progress"
    return LABEL_AMBIGUOUS, "ambiguous"


def counterfactual_repeatability(
    first,
    second,
    clearance_tolerance=0.05,
    progress_tolerance=0.03,
):
    """Check whether two identical-policy branches are stable enough to label."""
    _validate_outcome(first)
    _validate_outcome(second)
    if clearance_tolerance < 0.0 or progress_tolerance < 0.0:
        raise ValueError("repeatability tolerances must be non-negative")
    discrete_match = (
        bool(first["ego_collision"]) == bool(second["ego_collision"])
        and int(first["collision_count"]) == int(second["collision_count"])
        and bool(first["ego_target"]) == bool(second["ego_target"])
    )
    clearance_delta = abs(
        float(first["minimum_ego_clearance"])
        - float(second["minimum_ego_clearance"])
    )
    progress_delta = abs(float(first["ego_progress"]) - float(second["ego_progress"]))
    both_reached_target = bool(first["ego_target"]) and bool(second["ego_target"])
    return {
        "repeatable": bool(
            discrete_match
            and clearance_delta <= clearance_tolerance
            and (both_reached_target or progress_delta <= progress_tolerance)
        ),
        "discrete_match": bool(discrete_match),
        "clearance_delta": clearance_delta,
        "progress_delta": progress_delta,
    }
