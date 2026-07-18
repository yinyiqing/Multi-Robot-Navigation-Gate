import hashlib
import os


def build_eval_protocol_id(
    scenario_mode,
    eval_manifest_path,
    eval_episodes,
    max_episode_steps,
    manifest_sampling="cycle",
):
    parts = [
        "eval-v1",
        "scenario=%s" % scenario_mode,
        "episodes=%i" % int(eval_episodes),
        "max_steps=%i" % int(max_episode_steps),
    ]
    if eval_manifest_path:
        path = os.path.abspath(os.path.expanduser(eval_manifest_path))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        parts.extend(
            [
                "manifest_sha256=%s" % digest.hexdigest(),
                "sampling=%s" % manifest_sampling,
            ]
        )
    return "|".join(parts)


def reconcile_evaluation_state(checkpoint, current_protocol_id):
    if not checkpoint:
        return [], None, None, [], False, None

    evaluations = list(checkpoint.get("evaluations", []))
    best_summary = checkpoint.get("best_eval_summary")
    best_epoch = checkpoint.get("best_epoch")
    history = list(checkpoint.get("evaluation_history", []))
    saved_protocol_id = checkpoint.get("eval_protocol_id")
    if saved_protocol_id == current_protocol_id:
        return (
            evaluations,
            best_summary,
            best_epoch,
            history,
            False,
            saved_protocol_id,
        )

    if evaluations or best_summary is not None:
        history.append(
            {
                "eval_protocol_id": saved_protocol_id or "legacy-unversioned",
                "evaluations": evaluations,
                "best_eval_summary": best_summary,
                "best_epoch": best_epoch,
            }
        )
    return [], None, None, history, True, saved_protocol_id
