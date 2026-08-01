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


def _outcome_values(outcomes, key):
    if len(outcomes) < 2:
        raise ValueError("distribution labels require at least two rollouts per actor")
    for outcome in outcomes:
        _validate_outcome(outcome)
    return [float(outcome[key]) for outcome in outcomes]


def bootstrap_mean_difference_interval(
    generalist_values,
    strong_values,
    confidence=0.95,
    resamples=5000,
    seed=0,
):
    """Estimate a strong-minus-generalist mean difference interval."""
    import numpy as np

    generalist = np.asarray(generalist_values, dtype=np.float64)
    strong = np.asarray(strong_values, dtype=np.float64)
    if generalist.ndim != 1 or strong.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    if min(len(generalist), len(strong)) < 2:
        raise ValueError("bootstrap intervals require at least two values per actor")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not np.all(np.isfinite(generalist)) or not np.all(np.isfinite(strong)):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    generalist_indices = rng.integers(
        0, len(generalist), size=(resamples, len(generalist))
    )
    strong_indices = rng.integers(0, len(strong), size=(resamples, len(strong)))
    differences = (
        strong[strong_indices].mean(axis=1)
        - generalist[generalist_indices].mean(axis=1)
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(differences, [tail, 1.0 - tail])
    return {
        "generalist_mean": float(generalist.mean()),
        "strong_mean": float(strong.mean()),
        "mean_difference": float(strong.mean() - generalist.mean()),
        "interval": [float(low), float(high)],
        "confidence": float(confidence),
        "resamples": int(resamples),
    }


def choose_actor_distribution_label(
    generalist_outcomes,
    strong_outcomes,
    clearance_margin=0.10,
    progress_margin=0.05,
    maximum_progress_regression=0.05,
    confidence=0.95,
    resamples=5000,
    seed=0,
):
    """Choose an actor only when rollout distributions separate confidently."""
    if clearance_margin <= 0.0 or progress_margin <= 0.0:
        raise ValueError("counterfactual margins must be positive")
    if maximum_progress_regression < 0.0:
        raise ValueError("maximum_progress_regression must be non-negative")
    metric_keys = (
        "ego_collision",
        "collision_count",
        "ego_target",
        "minimum_ego_clearance",
        "ego_progress",
    )
    diagnostics = {}
    for index, key in enumerate(metric_keys):
        diagnostics[key] = bootstrap_mean_difference_interval(
            _outcome_values(generalist_outcomes, key),
            _outcome_values(strong_outcomes, key),
            confidence=confidence,
            resamples=resamples,
            seed=seed + index,
        )

    collision_interval = diagnostics["ego_collision"]["interval"]
    if collision_interval[1] < 0.0:
        return LABEL_STRONG, "ego_collision_rate", diagnostics
    if collision_interval[0] > 0.0:
        return LABEL_GENERALIST, "ego_collision_rate", diagnostics

    count_interval = diagnostics["collision_count"]["interval"]
    if count_interval[1] < 0.0:
        return LABEL_STRONG, "mean_collision_count", diagnostics
    if count_interval[0] > 0.0:
        return LABEL_GENERALIST, "mean_collision_count", diagnostics

    target_interval = diagnostics["ego_target"]["interval"]
    if target_interval[0] > 0.0:
        return LABEL_STRONG, "ego_target_rate", diagnostics
    if target_interval[1] < 0.0:
        return LABEL_GENERALIST, "ego_target_rate", diagnostics

    clearance_interval = diagnostics["minimum_ego_clearance"]["interval"]
    progress_interval = diagnostics["ego_progress"]["interval"]
    if (
        clearance_interval[0] >= clearance_margin
        and progress_interval[0] >= -maximum_progress_regression
    ):
        return LABEL_STRONG, "mean_clearance", diagnostics
    if (
        clearance_interval[1] <= -clearance_margin
        and progress_interval[1] <= maximum_progress_regression
    ):
        return LABEL_GENERALIST, "mean_clearance", diagnostics
    if (
        progress_interval[0] >= progress_margin
        and clearance_interval[0] >= -clearance_margin
    ):
        return LABEL_STRONG, "mean_progress", diagnostics
    if (
        progress_interval[1] <= -progress_margin
        and clearance_interval[1] <= clearance_margin
    ):
        return LABEL_GENERALIST, "mean_progress", diagnostics
    return LABEL_AMBIGUOUS, "ambiguous_distribution", diagnostics


def distribution_label_repeatability(labels):
    """Require independent rollout batches to agree on a non-ambiguous label."""
    labels = [int(label) for label in labels]
    stable = len(labels) >= 2 and labels[0] != LABEL_AMBIGUOUS
    repeated_label = (
        labels[0]
        if stable and all(label == labels[0] for label in labels)
        else LABEL_AMBIGUOUS
    )
    return {
        "repeatable": repeated_label != LABEL_AMBIGUOUS,
        "label": repeated_label,
        "batch_labels": labels,
    }
