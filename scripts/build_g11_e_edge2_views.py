#!/usr/bin/env python3
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAINLINE_DIR = (
    PROJECT_ROOT / "experiments/03_保留专门化/02_论文主线"
)
DEFAULT_SOURCE = MAINLINE_DIR / "08_exact_edge2零样本确认/validation.json"
DEFAULT_OUTPUT_DIR = (
    MAINLINE_DIR / "datasets/fixed_v1/views/g11_e_edge2_generalization_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze G11-E exact-edge-2 pilot and confirmation views."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-scenarios", type=int, default=50)
    return parser.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(encoded)


def annotate(source, partition):
    scenario = copy.deepcopy(source)
    scenario["navigation_split"] = "validation"
    scenario["split"] = "validation"
    scenario["view"] = {
        **scenario.get("view", {}),
        "gate_protocol": "g11-e-edge2-v1",
        "gate_topology": "edge2",
        "g11_e_partition": partition,
    }
    return scenario


def make_payload(
    source, scenarios, partition, pilot_scenarios, source_path, source_sha256
):
    payload = dict(source)
    payload.update(
        {
            "dataset_id": "g11_e_edge2_generalization_v1_%s" % partition,
            "split": "validation",
            "view_config": {
                "gate_protocol": "g11-e-edge2-v1",
                "partition": partition,
                "source": str(source_path),
                "source_sha256": source_sha256,
                "selection": "frozen_source_order",
                "pilot_scenarios": pilot_scenarios,
                "sealed_test_read": False,
            },
            "scenarios": [annotate(item, partition) for item in scenarios],
        }
    )
    return payload


def build_views(source_path, output_dir, pilot_scenarios=50):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    scenarios = source.get("scenarios") or []
    if len(scenarios) != 200:
        raise ValueError("G11-E source must contain exactly 200 scenarios")
    if not 0 < int(pilot_scenarios) < len(scenarios):
        raise ValueError("pilot_scenarios must split the source into two nonempty views")
    ids = [str(item["scenario_id"]) for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("G11-E source contains duplicate scenario IDs")
    if any(int(item.get("metrics", {}).get("conflict_edge_count", -1)) != 2 for item in scenarios):
        raise ValueError("G11-E source contains a non-edge-2 scenario")

    source_sha256 = sha256_file(source_path)
    try:
        source_label = source_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        source_label = source_path.resolve()
    parts = {
        "pilot": scenarios[:pilot_scenarios],
        "confirmation": scenarios[pilot_scenarios:],
    }
    outputs = {}
    for partition, selected in parts.items():
        output = output_dir / (partition + ".json.gz")
        payload = make_payload(
            source,
            selected,
            partition,
            int(pilot_scenarios),
            source_label,
            source_sha256,
        )
        write_gzip_json(output, payload)
        outputs[partition] = {
            "path": str(output),
            "scenarios": len(selected),
            "sha256": sha256_file(output),
        }
    return outputs


def main():
    args = parse_args()
    print(
        json.dumps(
            build_views(args.source, args.output_dir, args.pilot_scenarios),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
