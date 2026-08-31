"""Static Geometry-V5 M0 contract checker; it never executes a real model."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from cegwm.protocol.geometry_v5_m0 import load_geometry_v5_m0_contract
from cegwm.protocol.geometry_v5_m0_colab import load_m0_execution_contract
from cegwm.runtime.geometry_v5_m0_sd21 import public_runtime_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("development",), default="development")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--expected-exact")
    parser.add_argument("--validate-static", action="store_true", help="validate byte-bound M0 config and manifest only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_geometry_v5_m0_contract(args.repo_root)
    execution = load_m0_execution_contract(args.repo_root)
    if len(contract.units) * len(contract.config["development"]["attacks"]) != 44:
        raise RuntimeError("M0 physical denominator differs")
    if args.validate_static:
        print(execution.config["claim_ceiling"])
        return 0
    if args.artifact_root is None or not args.expected_exact:
        raise SystemExit("real development requires --artifact-root and --expected-exact")
    _validate_real_preflight(args.repo_root, args.expected_exact)
    if args.artifact_root.exists():
        raise RuntimeError("artifact root must be create-only and absent")
    from cegwm.runtime.geometry_v5_m0_sd21 import load_bound_sd21_pipeline

    del load_bound_sd21_pipeline
    raise RuntimeError("real M0 execution requires a separately supplied Colab runtime adapter")


def _validate_real_preflight(repo_root: Path, expected_exact: str) -> None:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()

    if git("rev-parse", "HEAD") != expected_exact or git("status", "--porcelain"):
        raise RuntimeError("real execution exact or clean state differs")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo_root).returncode == 0:
        raise RuntimeError("real execution requires detached HEAD")


if __name__ == "__main__":
    raise SystemExit(main())
