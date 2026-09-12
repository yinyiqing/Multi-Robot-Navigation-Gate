#!/usr/bin/env python3
import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from g27_counterfactual import BRANCH_NAMES, analyze_pilot, canonical_sha256


BASE = ROOT / "experiments/03_保留专门化/02_论文主线"
DEFAULT_ANCHORS = BASE / "27_反事实Reward增强Gate监督/local_data/protocol/p0_anchors.json"
DEFAULT_MANIFEST = BASE / "datasets/fixed_v1/views/g11_a1_gate_v1/train.json.gz"
DEFAULT_NAVIGATION = (
    ROOT
    / "TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best_actor.pth"
)
DEFAULT_INTERACTION = (
    ROOT
    / "TD3/pytorch_models/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016_actor.pth"
)
WORKER = ROOT / "TD3/run_g27_counterfactual_branch.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate isolated reference and N1/N2/I1/I2 G27 processes."
    )
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--navigation-actor", type=Path, default=DEFAULT_NAVIGATION)
    parser.add_argument("--interaction-actor", type=Path, default=DEFAULT_INTERACTION)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "collect"), required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-ros-port", type=int, default=18110)
    parser.add_argument("--base-gazebo-port", type=int, default=19110)
    parser.add_argument("--process-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            handle.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def wait_for_ports(ports, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(port_is_free(port) for port in ports):
            return
        time.sleep(0.5)
    raise RuntimeError("ROS/Gazebo ports were not released: %s" % (ports,))


def stop_process_group(process, ports):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        wait_for_ports(ports, timeout=5.0)
        return
    except RuntimeError:
        pass
    targets = ["%d/tcp" % port for port in ports]
    subprocess.run(
        ["fuser", "-k", "-TERM"] + targets,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        wait_for_ports(ports, timeout=8.0)
        return
    except RuntimeError:
        pass
    subprocess.run(
        ["fuser", "-k", "-KILL"] + targets,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    wait_for_ports(ports, timeout=10.0)


def run_worker(args, slot, mode, input_path, output_path, log_path, branch=None):
    ros_port = args.base_ros_port + slot
    gazebo_port = args.base_gazebo_port + slot
    ports = (ros_port, gazebo_port)
    command = [
        sys.executable,
        "-u",
        str(WORKER),
        "--mode",
        mode,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--manifest",
        str(args.manifest),
        "--navigation-actor",
        str(args.navigation_actor),
        "--interaction-actor",
        str(args.interaction_actor),
        "--seed",
        str(args.seed),
    ]
    if branch is not None:
        command.extend(("--branch", branch))
    if args.detector_checkpoint is not None:
        command.extend(("--detector-checkpoint", str(args.detector_checkpoint)))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "ROS_HOSTNAME": "localhost",
            "ROS_MASTER_URI": "http://localhost:%d" % ros_port,
            "ROS_PORT_SIM": str(ros_port),
            "GAZEBO_MASTER_URI": "http://localhost:%d" % gazebo_port,
            "GAZEBO_IP": "127.0.0.1",
            "GAZEBO_RESOURCE_PATH": str(
                ROOT / "catkin_ws/src/multi_robot_scenario/launch"
            ),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for attempt in range(1, 4):
        attempt_log = (
            log_path
            if attempt == 1
            else log_path.with_name(
                "%s.attempt%d%s" % (log_path.stem, attempt, log_path.suffix)
            )
        )
        try:
            wait_for_ports(ports, timeout=5.0)
            with attempt_log.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT / "TD3",
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                try:
                    return_code = process.wait(timeout=args.process_timeout)
                except subprocess.TimeoutExpired:
                    return_code = -1
                finally:
                    stop_process_group(process, ports)
            if return_code == 0 and output_path.is_file():
                return
            tail = attempt_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-30:]
            failures.append(
                "attempt %d code %d (%s):\n%s"
                % (attempt, return_code, attempt_log, "\n".join(tail))
            )
        except Exception as error:
            failures.append("attempt %d infrastructure error: %s" % (attempt, error))
        if output_path.exists():
            raise RuntimeError(
                "worker left an output after a failed exit; refusing to overwrite %s"
                % output_path
            )
    raise RuntimeError("worker failed after 3 attempts:\n%s" % "\n".join(failures))


def run_anchor(args, anchor, slot):
    anchor_dir = args.output_dir / "anchors" / anchor["anchor_id"]
    anchor_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = anchor_dir / "candidate.json"
    reference_path = anchor_dir / "reference.json"
    if candidate_path.exists():
        if load_json(candidate_path) != anchor:
            raise RuntimeError("saved candidate differs from frozen anchor selection")
    else:
        write_json(candidate_path, anchor)
    if not reference_path.exists():
        run_worker(
            args,
            slot,
            "reference",
            candidate_path,
            reference_path,
            anchor_dir / "reference.log",
        )
    reference = load_json(reference_path)
    if not reference.get("valid", True):
        combined = {
            "anchor_id": anchor["anchor_id"],
            "candidate": anchor,
            "reference": reference,
            "branches": {},
            "status": "invalid_reference",
        }
        combined["record_sha256"] = canonical_sha256(combined)
        write_json(anchor_dir / "combined.json", combined)
        return combined
    branches = {}
    for branch in BRANCH_NAMES:
        output_path = anchor_dir / (branch + ".json")
        if not output_path.exists():
            run_worker(
                args,
                slot,
                "branch",
                reference_path,
                output_path,
                anchor_dir / (branch + ".log"),
                branch=branch,
            )
        result = load_json(output_path)
        if result.get("spec_sha256") != reference.get("spec_sha256"):
            raise RuntimeError("branch/reference hash mismatch for %s" % anchor["anchor_id"])
        branches[branch] = result
    combined = {
        "anchor_id": anchor["anchor_id"],
        "candidate": anchor,
        "reference": reference,
        "branches": branches,
        "status": "complete",
    }
    combined["record_sha256"] = canonical_sha256(combined)
    write_json(anchor_dir / "combined.json", combined)
    return combined


def validate_inputs(args):
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.process_timeout <= 0.0:
        raise ValueError("process timeout must be positive")
    for path in (
        args.anchors,
        args.manifest,
        args.navigation_actor,
        args.interaction_actor,
        WORKER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.detector_checkpoint is not None and not args.detector_checkpoint.is_file():
        raise FileNotFoundError(args.detector_checkpoint)
    ports = []
    for slot in range(args.jobs):
        ports.extend((args.base_ros_port + slot, args.base_gazebo_port + slot))
    if len(ports) != len(set(ports)) or not all(port_is_free(port) for port in ports):
        raise RuntimeError("requested ROS/Gazebo port pool is not free and unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            "refusing to reuse a non-empty run directory without --resume: %s"
            % args.output_dir
        )


def main():
    args = parse_args()
    validate_inputs(args)
    selection = load_json(args.anchors)
    anchors = []
    for item in selection["anchors"]:
        anchor = dict(item)
        anchor["protocol"] = selection["protocol"]
        anchors.append(anchor)
    if args.limit is not None:
        anchors = anchors[: args.limit]
    if args.profile == "smoke" and len(anchors) != 1:
        raise ValueError("smoke profile requires exactly one anchor (--limit 1)")
    if args.profile == "pilot" and len(anchors) != 32:
        raise ValueError("P0 pilot requires all 32 frozen anchors")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "format_version": 1,
        "profile": args.profile,
        "selection_sha256": selection["selection_sha256"],
        "anchor_ids": [item["anchor_id"] for item in anchors],
        "jobs": args.jobs,
        "base_ros_port": args.base_ros_port,
        "base_gazebo_port": args.base_gazebo_port,
        "process_timeout_seconds": args.process_timeout,
        "seed": args.seed,
        "inputs": {
            "manifest_sha256": file_sha256(args.manifest),
            "navigation_actor_sha256": file_sha256(args.navigation_actor),
            "interaction_actor_sha256": file_sha256(args.interaction_actor),
            "detector_sha256": (
                file_sha256(args.detector_checkpoint)
                if args.detector_checkpoint is not None
                else None
            ),
        },
    }
    run_manifest["run_manifest_sha256"] = canonical_sha256(run_manifest)
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing != run_manifest:
            raise RuntimeError("resume run manifest does not match frozen configuration")
    else:
        write_json(manifest_path, run_manifest)

    slots = queue.Queue()
    for slot in range(args.jobs):
        slots.put(slot)

    def scheduled(anchor):
        slot = slots.get()
        try:
            return run_anchor(args, anchor, slot)
        finally:
            slots.put(slot)

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(scheduled, anchor): anchor for anchor in anchors}
        try:
            for future in as_completed(futures):
                anchor = futures[future]
                record = future.result()
                records.append(record)
                if record["status"] == "invalid_reference":
                    print(
                        "Completed %s (invalid reference: %s)"
                        % (
                            anchor["anchor_id"],
                            record["reference"]["invalid_reason"],
                        ),
                        flush=True,
                    )
                else:
                    passed = sum(
                        int(record["branches"][name]["alignment"]["passed"])
                        for name in BRANCH_NAMES
                    )
                    print(
                        "Completed %s (%d/4 aligned branches)"
                        % (anchor["anchor_id"], passed),
                        flush=True,
                    )
        except Exception:
            for pending in futures:
                pending.cancel()
            raise
    order = {anchor["anchor_id"]: index for index, anchor in enumerate(anchors)}
    records.sort(key=lambda item: order[item["anchor_id"]])
    if args.profile == "pilot":
        analysis = analyze_pilot(records)
    elif args.profile == "smoke":
        analysis = {
            "format_version": 1,
            "profile": "smoke",
            "anchors_total": 1,
            "branches_aligned": sum(
                int(records[0]["branches"][name]["alignment"]["passed"])
                for name in BRANCH_NAMES
            ),
            "all_branches_aligned": all(
                records[0]["branches"][name]["alignment"]["passed"]
                for name in BRANCH_NAMES
            ),
            "reward_environment_matches": True,
            "passed": all(
                records[0]["branches"][name]["alignment"]["passed"]
                for name in BRANCH_NAMES
            ),
        }
    else:
        valid = [item for item in records if item["status"] == "complete"]
        aligned = [
            item
            for item in valid
            if all(
                item["branches"][name]["alignment"]["passed"]
                for name in BRANCH_NAMES
            )
        ]
        analysis = {
            "format_version": 1,
            "profile": "collect",
            "anchors_total": len(records),
            "reference_valid": len(valid),
            "reference_invalid": len(records) - len(valid),
            "all_branches_aligned": len(aligned),
            "excluded_for_alignment": len(valid) - len(aligned),
            "usable_anchors": len(aligned),
            "completed": True,
            "passed": True,
        }
    summary = {
        "run_manifest": run_manifest,
        "analysis": analysis,
        "anchor_record_sha256": {
            item["anchor_id"]: item["record_sha256"] for item in records
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("Summary:", args.output_dir / "summary.json")
    if not analysis["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
