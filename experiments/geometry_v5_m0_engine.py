"""Static Geometry-V5 M0 contract checker; it never executes a real model."""

from __future__ import annotations

import argparse
from pathlib import Path

from cegwm.protocol.geometry_v5_m0 import load_geometry_v5_m0_contract
from cegwm.runtime.geometry_v5_m0_sd21 import public_runtime_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate-static", action="store_true", help="validate byte-bound M0 config and manifest only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.validate_static:
        raise SystemExit("M0 engine is static-only; pass --validate-static")
    contract = load_geometry_v5_m0_contract(args.repo_root)
    if len(contract.units) * len(contract.config["development"]["attacks"]) != 44:
        raise RuntimeError("M0 physical denominator differs")
    if public_runtime_capabilities()["real_model_adapter_bound"]:
        raise RuntimeError("M0 static engine must not bind a real model adapter")
    print(contract.config["engineering_evaluation"]["claim_ceiling"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
