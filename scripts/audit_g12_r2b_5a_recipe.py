#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TD3"))

from actor_models import Actor, function_preserving_expand_actor_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-actor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(20260823)
    source_state = torch.load(args.source_actor, map_location="cpu")
    source = Actor(24, 2)
    source.load_state_dict(source_state)
    wide = Actor(24, 2, hidden_dim_1=1137, hidden_dim_2=855)
    wide.load_state_dict(function_preserving_expand_actor_state_dict(source_state, wide))

    states = torch.randn(1024, 24)
    with torch.no_grad():
        max_output_error = float((wide(states) - source(states)).abs().max())

    wide.zero_grad(set_to_none=True)
    wide(states[:128]).square().mean().backward()
    added_gradient = float(wide.layer_3.weight.grad[:, 600:].abs().sum())
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    wide_parameters = sum(parameter.numel() for parameter in wide.parameters())
    expected_wide_parameters = 1_003_127
    passed = (
        max_output_error <= 1e-5
        and added_gradient > 0
        and source_parameters == 501_802
        and wide_parameters == expected_wide_parameters
    )
    result = {
        "source_actor": str(args.source_actor),
        "source_architecture": [24, 800, 600, 2],
        "wide_architecture": [24, 1137, 855, 2],
        "source_parameters": source_parameters,
        "wide_parameters": wide_parameters,
        "max_output_error": max_output_error,
        "added_output_gradient_l1": added_gradient,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("R2B initialization audit failed")


if __name__ == "__main__":
    main()
