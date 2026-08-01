from collections import defaultdict

import numpy as np


def discounted_n_step_target(rewards, discount, bootstrap_value=0.0):
    """Return the discounted rewards followed by an optional bootstrap value."""
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    target = float(bootstrap_value)
    for reward in reversed(tuple(rewards)):
        target = float(reward) + discount * target
    return target


def pairwise_order_counts(predicted, observed, minimum_observed_gap=1e-3):
    """Count within-state action pairs whose predicted and observed orders agree."""
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")
    if predicted.ndim != 1:
        raise ValueError("predicted and observed must be one-dimensional")
    if minimum_observed_gap < 0.0:
        raise ValueError("minimum_observed_gap must be non-negative")

    comparable = 0
    agreements = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            observed_delta = observed[left] - observed[right]
            if abs(observed_delta) <= minimum_observed_gap:
                continue
            comparable += 1
            predicted_delta = predicted[left] - predicted[right]
            agreements += int(predicted_delta * observed_delta > 0.0)
    return agreements, comparable


def summarize_counterfactual_calibration(records, minimum_observed_gap=1e-3):
    """Summarize Q calibration without mixing actions from different states."""
    grouped = defaultdict(list)
    repeatable_records = []
    for record in records:
        if not record.get("repeatable", False):
            continue
        key = (
            record["scenario_id"],
            int(record["anchor_step"]),
            int(record["ego_index"]),
        )
        grouped[key].append(record)
        repeatable_records.append(record)

    agreements = 0
    comparable = 0
    calibrated_groups = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        group_agreements, group_comparable = pairwise_order_counts(
            [item["predicted_qmin"] for item in group],
            [item["observed_n_step_target"] for item in group],
            minimum_observed_gap,
        )
        if group_comparable:
            calibrated_groups += 1
        agreements += group_agreements
        comparable += group_comparable

    errors = [
        float(item["predicted_qmin"] - item["observed_n_step_target"])
        for item in repeatable_records
    ]
    return {
        "records": len(records),
        "repeatable_records": len(repeatable_records),
        "state_action_groups": len(grouped),
        "calibrated_groups": calibrated_groups,
        "pairwise_comparisons": comparable,
        "pairwise_order_agreements": agreements,
        "pairwise_order_accuracy": agreements / comparable if comparable else None,
        "qmin_target_mae": float(np.mean(np.abs(errors))) if errors else None,
        "qmin_target_bias": float(np.mean(errors)) if errors else None,
    }
