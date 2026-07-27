from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    threshold: float
    precision: float
    recall: float
    fpr: float
    proposal_recall: float
    f1: float
    true_positive: int
    false_positive: int
    negative_candidates: int
    positive_candidates: int
    visible_robots: int
    missed_visible_robots: int
    meets_entry_criteria: bool

    def to_dict(self):
        return asdict(self)


def detection_metrics(probabilities, labels, visible_robot_count, threshold=0.5):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    if probabilities.shape != labels.shape or probabilities.ndim != 1:
        raise ValueError("probabilities and labels must be matching vectors")
    visible = int(visible_robot_count)
    if visible < int(np.sum(labels)):
        raise ValueError("visible_robot_count cannot be smaller than positive proposals")
    predictions = probabilities >= float(threshold)
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    positive_candidates = int(np.sum(positives))
    negative_candidates = int(np.sum(negatives))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(visible, 1)
    fpr = false_positive / max(negative_candidates, 1)
    proposal_recall = positive_candidates / max(visible, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return DetectionMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        fpr=float(fpr),
        proposal_recall=float(proposal_recall),
        f1=float(f1),
        true_positive=true_positive,
        false_positive=false_positive,
        negative_candidates=negative_candidates,
        positive_candidates=positive_candidates,
        visible_robots=visible,
        missed_visible_robots=max(visible - positive_candidates, 0),
        meets_entry_criteria=bool(
            precision >= 0.70 and recall >= 0.90 and fpr <= 0.10
        ),
    )


def select_validation_threshold(probabilities, labels, visible_robot_count):
    candidates = np.linspace(0.05, 0.95, 91)
    metrics = [
        detection_metrics(probabilities, labels, visible_robot_count, threshold)
        for threshold in candidates
    ]
    feasible = [item for item in metrics if item.meets_entry_criteria]
    if feasible:
        return max(feasible, key=lambda item: (item.precision, item.f1, item.threshold))
    return max(metrics, key=lambda item: (item.f1, item.recall, -item.fpr))
