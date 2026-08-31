"""Local static smoke for Geometry-V5 M0; it never loads or executes a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from cegwm.protocol.geometry_v5_m0 import load_geometry_v5_m0_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("development",), default="development")
    parser.add_argument("--validate-static", action="store_true", help="validate the base M0 contract only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_geometry_v5_m0_contract(args.repo_root)
    template = contract.config["template"]
    if template["channel"] != 3 or template["scale"] != 5 or tuple(template["radial_lengths"]) != (0.2, 0.3, 0.4, 0.5):
        raise RuntimeError("M0 static template identity differs")
    if tuple(unit.seed for unit in contract.units) != (7501, 7502, 7503, 7504):
        raise RuntimeError("M0 static method identity differs")
    print(contract.config["engineering_evaluation"]["claim_ceiling"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
