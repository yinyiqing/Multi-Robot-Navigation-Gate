def actor_gradient_gate_decision(
    safety_samples,
    linear_positive_share,
    angular_positive_share,
    min_samples,
    max_linear_positive_share,
    max_angular_one_sided_share,
):
    angular_one_sided_share = max(
        angular_positive_share, 1.0 - angular_positive_share
    )
    passed = (
        safety_samples >= min_samples
        and linear_positive_share <= max_linear_positive_share
        and angular_one_sided_share <= max_angular_one_sided_share
    )
    return bool(passed), float(angular_one_sided_share)
