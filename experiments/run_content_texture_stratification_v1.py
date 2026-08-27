"""Prospective, analysis-only Content texture stratification coordinator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from cegwm.protocol.content_texture_stratification_v1 import (
    exact_spearman,
    f64_hex,
    load_protocol,
    margins,
    median,
    parse_p6_texture,
    require_scores,
    sha256_bytes,
    stable_json_bytes,
    stratified_exact,
)

KEY_ENV = "CEG_WM_ROOT_KEY"
TOKEN_ENV = "HF_TOKEN"
EVENT_PREFIX = "CEGWM_TEXTURE_EVENT "
RESULT_PREFIX = "CEGWM_TEXTURE_RESULT"
PUBLIC_FAILURES = {"FileExistsError", "FileNotFoundError", "ImportError", "MemoryError", "OSError", "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError"}
FAILURE_STAGES = frozenset({
    "identity", "protocol", "secrets", "checkouts", "rosters", "assets",
    "prefetch", "common_plain", "v2", "v3", "v4", "v5_validate", "v6",
    "v7", "v8", "analysis", "terminal_publication",
})
_failure_stage = "identity"
METHOD_ORDER = ("v2", "v3", "v4", "v5", "v6", "v7", "v8")
SCORE_FIELDS = tuple(f"{branch}__{label}" for branch in ("lf", "hf", "joint") for label in ("registered", *(f"wrong_{index:02d}" for index in range(16))))
OPERATIONAL_RESULT_FIELDS = ("artifact_kind", "status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256", "failure_class", "failure_stage", "last_completed_checkpoint", "result_member")
_failure_context: dict[str, Any] | None = None
PER_UNIT_COLUMNS = (
    "global_ordinal", "roster_id", "roster_ordinal", "unit_id", "source_id", "seed", "method_id", "source_exact", "lf_score_domain", "status", "failure_class", "plain_ppm_sha256", "plain_rgb_sha256", "texture_value", "texture_be_hex", "texture_rank", "texture_rank_be_hex", "candidate_rgb_sha256", "primary_null_rgb_sha256", "primary_null_matches_plain", "lf_registered", "lf_max_wrong", "lf_null_registered", "lf_margin_a", "lf_margin_b", "hf_registered", "hf_max_wrong", "hf_null_registered", "hf_margin_a", "hf_margin_b", "joint_or_identity", "reuse_source_method", "missing_note",
)
ASSOCIATION_COLUMNS = (
    "method_id", "source_exact", "lf_score_domain", "branch_role", "branch", "margin_id", "scope", "roster_id", "fixed_n", "observed_pair_count", "statistic_id", "statistic_value", "statistic_be_hex", "c_numerator", "c_denominator", "permutation_scheme", "permutation_extreme_count", "permutation_total_count", "permutation_p_value", "permutation_p_be_hex", "texture_median", "margin_median", "rho_old", "rho_current", "rho_difference", "same_nonzero_sign", "interpretability", "missing_unit_ids", "note",
)


def _set_failure_stage(stage: str) -> None:
    global _failure_stage
    if stage not in FAILURE_STAGES:
        raise ValueError("texture failure stage differs")
    _failure_stage = stage


def _failure_line(error: BaseException) -> str:
    name = type(error).__name__
    payload = {
        "status": "analysis_incomplete",
        "failure_class": name if name in PUBLIC_FAILURES else "OtherOperationalError",
        "failure_stage": _failure_stage,
    }
    line = f"{RESULT_PREFIX} " + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(line.encode("utf-8")) > 4096:
        raise RuntimeError("texture failure diagnostic exceeds bound")
    return line


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _identity(repo: Path, expected: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None or _git(repo, "rev-parse", "HEAD") != expected or _git(repo, "status", "--porcelain"):
        raise RuntimeError("analysis checkout identity differs")


def _load_roster(path: Path, expected_sha: str, roster_id: str, offset: int) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise RuntimeError("roster bytes differ")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 8:
        raise ValueError("texture roster must contain exactly eight units")
    units = []
    for ordinal, row in enumerate(rows, 1):
        if set(row) != {"unit_id", "split", "source_id", "prompt", "seed", "height", "width"} or row["height"] != 512 or row["width"] != 512:
            raise ValueError("texture roster unit schema differs")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise TypeError("texture seed differs")
        units.append({"global_ordinal": offset + ordinal, "roster_id": roster_id, "roster_ordinal": ordinal, **row})
    return units


def _create_checkouts(repo: Path, checkouts: Path, sources: Mapping[str, Any]) -> dict[str, Path]:
    checkouts.mkdir(parents=True)
    resolved = {}
    for name, identity in sources.items():
        path = checkouts / name
        subprocess.run(["git", "worktree", "add", "--detach", str(path), identity["exact"]], cwd=repo, check=True, capture_output=True, text=True)
        if _git(path, "rev-parse", "HEAD") != identity["exact"] or _git(path, "rev-parse", "HEAD^{tree}") != identity["tree"] or _git(path, "status", "--porcelain"):
            raise RuntimeError("detached source checkout differs")
        resolved[name] = path
    return resolved


def _stage_asset(provenance_root: Path, output: Path, spec: Mapping[str, Any]) -> Path:
    source = provenance_root / spec["provenance_zip"]
    if sha256_bytes(source.read_bytes()) != spec["zip_sha256"]:
        raise RuntimeError("provenance archive hash differs")
    output.mkdir(parents=True)
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("provenance archive CRC differs")
        payload = archive.read(spec["member"])
        sidecar = archive.read(spec["sidecar_member"])
    if sha256_bytes(payload) != spec["sha256"] or sha256_bytes(sidecar) != spec["sidecar_file_sha256"]:
        raise RuntimeError("staged public asset hash differs")
    filename = Path(spec["member"]).name
    if sidecar != f"{spec['sha256']}  {filename}\n".encode("ascii"):
        raise RuntimeError("staged public asset sidecar differs")
    with (output / filename).open("xb") as handle:
        handle.write(payload)
    with (output / f"{filename}.sha256").open("xb") as handle:
        handle.write(sidecar)
    return output


def _v4_blob_bindings(v4: Path, v5: Path) -> dict[str, str]:
    paths = ("experiments/run_content_v4_clean.py", "src/cegwm/method/content_whitening_v4.py", "src/cegwm/runtime/content_adaptive_sd35_v3.py")
    values = {}
    for relative in paths:
        left, right = (v4 / relative).read_bytes(), (v5 / relative).read_bytes()
        if left != right:
            raise RuntimeError("V5 delegated V4 production blob differs")
        values[relative] = sha256_bytes(left)
    return values


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(stable_json_bytes(value))


def _checkpoint(root: Path, index: int, identity: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    payload = stable_json_bytes({"schema_version": "content_texture_stratification_checkpoint_v1", "identity": dict(identity), "checkpoint_index": index, "resume_allowed": False, "state": dict(state)})
    path = root / f"checkpoint-{index:04d}.json"
    sidecar = root / f"checkpoint-{index:04d}.json.sha256"
    json_created = False
    sidecar_created = False
    try:
        handle = path.open("xb")
        json_created = True
        with handle:
            handle.write(payload)
            handle.flush()
        digest = sha256_bytes(payload)
        handle = sidecar.open("xb")
        sidecar_created = True
        with handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
    except BaseException:
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        if json_created:
            path.unlink(missing_ok=True)
        raise


def _checkpoint_event(event: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: value for name, value in event.items() if name not in {"scores", "primary_null_scores"}}
    for group in ("scores", "primary_null_scores"):
        if group in event:
            result[f"{group}_be_hex"] = {name: f64_hex(float(value)) for name, value in event[group].items()}
    return result


def _ordered_scores(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) != len(SCORE_FIELDS) or set(value) != set(SCORE_FIELDS):
        raise ValueError("score fields differ")
    ordered = {name: value[name] for name in SCORE_FIELDS}
    require_scores(ordered)
    return ordered


def _adapter_event(line: str) -> dict[str, Any]:
    event = json.loads(line[len(EVENT_PREFIX):])
    if not isinstance(event, dict):
        raise ValueError("adapter event must be an object")
    for name in ("scores", "primary_null_scores"):
        if name in event:
            event[name] = _ordered_scores(event[name])
    return event


def _child(adapter: Path, source: Path, exact: str, phase: str, units_path: Path, output: Path, cache: Path, bindings_path: Path | None, env: Mapping[str, str], *, v7_asset: Path | None = None, v8_asset: Path | None = None) -> list[dict[str, Any]]:
    command = [sys.executable, str(adapter), "--source-root", str(source), "--expected-exact", exact, "--phase", phase, "--units-json", str(units_path), "--plain-bindings-json", str(output / "plain_bindings.json"), "--local-output-root", str(output), "--hf-cache-root", str(cache)]
    if bindings_path is not None:
        command += ["--model-bindings-json", str(bindings_path)]
    if v7_asset is not None:
        command += ["--v7-asset-root", str(v7_asset)]
    if v8_asset is not None:
        command += ["--v8-asset-root", str(v8_asset)]
    child_env = {**os.environ, **env, "HF_HOME": str(cache)}
    process = subprocess.Popen(command, cwd=source, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="strict")
    events = []
    assert process.stdout is not None
    for line in process.stdout:
        if len(line) > 65536:
            process.kill()
            raise RuntimeError("adapter output line exceeds bound")
        if line.startswith(EVENT_PREFIX):
            events.append(_adapter_event(line))
            if len(events) > 64:
                process.kill()
                raise RuntimeError("adapter event count exceeds bound")
    rc = process.wait()
    if rc != 0 or not events or events[-1] != {"event": "phase_complete", "phase": phase}:
        raise RuntimeError("adapter phase failed or ended out of order")
    return events


def _unit_events(events: Sequence[Mapping[str, Any]], method: str, expected: int) -> list[dict[str, Any]]:
    selected = [dict(item) for item in events if item.get("event") == ("plain" if method == "common_plain" else "unit")]
    if len(selected) != expected or [item["global_ordinal"] for item in selected] != list(range(1, expected + 1)):
        raise RuntimeError("adapter unit event order differs")
    if method != "common_plain" and any(item.get("method") != method for item in selected):
        raise RuntimeError("adapter method event identity differs")
    return selected


def _derive_row(unit: Mapping[str, Any], method: str, source: Mapping[str, Any], plain: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    base = {name: "" for name in PER_UNIT_COLUMNS}
    base.update({"global_ordinal": unit["global_ordinal"], "roster_id": unit["roster_id"], "roster_ordinal": unit["roster_ordinal"], "unit_id": unit["unit_id"], "source_id": unit["source_id"], "seed": unit["seed"], "method_id": source["method_id"], "source_exact": source["exact"], "lf_score_domain": source["lf_score_domain"], "status": event["status"], "failure_class": event.get("failure_class", ""), "plain_ppm_sha256": plain.get("plain_ppm_sha256", ""), "plain_rgb_sha256": plain.get("plain_rgb_sha256", ""), "joint_or_identity": "branchwise_or_derived_only" if method == "v5" else "joint_min_recorded_not_analyzed", "reuse_source_method": source.get("reuse_source_method", "")})
    if event["status"] != "success" or plain.get("status") != "success":
        base["missing_note"] = "retained_unit_or_plain_failure"
        return base
    scores, null_scores = event["scores"], event["primary_null_scores"]
    lf_a, lf_b = margins(scores, null_scores, "lf")
    hf_a, hf_b = margins(scores, null_scores, "hf")
    texture = float(plain["texture_value"])
    base.update({"texture_value": format(texture, ".17g"), "texture_be_hex": f64_hex(texture), "candidate_rgb_sha256": event["candidate_rgb_sha256"], "primary_null_rgb_sha256": event["primary_null_rgb_sha256"], "primary_null_matches_plain": event["primary_null_rgb_sha256"] == plain["plain_rgb_sha256"], "lf_registered": format(float(scores["lf__registered"]), ".17g"), "lf_max_wrong": format(max(float(scores[f"lf__wrong_{i:02d}"]) for i in range(16)), ".17g"), "lf_null_registered": format(float(null_scores["lf__registered"]), ".17g"), "lf_margin_a": format(lf_a, ".17g"), "lf_margin_b": format(lf_b, ".17g"), "hf_registered": format(float(scores["hf__registered"]), ".17g"), "hf_max_wrong": format(max(float(scores[f"hf__wrong_{i:02d}"]) for i in range(16)), ".17g"), "hf_null_registered": format(float(null_scores["hf__registered"]), ".17g"), "hf_margin_a": format(hf_a, ".17g"), "hf_margin_b": format(hf_b, ".17g")})
    if not base["primary_null_matches_plain"]:
        base["missing_note"] = "method_primary_null_differs_from_dedicated_plain"
    return base


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    from cegwm.protocol.content_texture_stratification_v1 import average_ranks
    plain_by_unit: dict[tuple[str, str], tuple[str, str]] = {}
    for roster in ("content_v234_old", "content_v6_current"):
        subset = [row for row in rows if row["roster_id"] == roster and row["method_id"] == rows_by_method(rows)[METHOD_ORDER[0]][0]["method_id"]]
        if len(subset) != 8 or any(not row["texture_value"] for row in subset):
            continue
        ranks = average_ranks([float(row["texture_value"]) for row in subset])
        for row, rank in zip(subset, ranks):
            plain_by_unit[(roster, row["unit_id"])] = (str(float(rank)), f64_hex(float(rank)))
    for row in rows:
        rank = plain_by_unit.get((row["roster_id"], row["unit_id"]))
        if rank:
            row["texture_rank"], row["texture_rank_be_hex"] = rank


def rows_by_method(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in METHOD_ORDER}
    for name in METHOD_ORDER:
        result[name] = rows[name_index(name) * 16:(name_index(name) + 1) * 16]
    return result


def name_index(name: str) -> int:
    return METHOD_ORDER.index(name)


def _association_rows(rows: list[dict[str, Any]], sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    grouped = rows_by_method(rows)
    for method in METHOD_ORDER:
        method_rows = grouped[method]
        for branch in ("lf", "hf"):
            for margin_id in ("a", "b"):
                roster_stats: dict[str, dict[str, Any]] = {}
                for roster in ("content_v234_old", "content_v6_current"):
                    subset = [item for item in method_rows if item["roster_id"] == roster]
                    available = [item for item in subset if not item["missing_note"] and item[f"{branch}_margin_{margin_id}"]]
                    stat = exact_spearman([float(item["texture_value"]) for item in available], [float(item[f"{branch}_margin_{margin_id}"]) for item in available]) if len(available) == 8 else {"interpretability": "unavailable_incomplete_fixed_denominator"}
                    roster_stats[roster] = stat
                    result.append(_association(source=sources[method], branch=branch, margin_id=margin_id, scope="within_roster", roster_id=roster, fixed_n=8, subset=subset, available=available, stat=stat))
                old = [item for item in method_rows if item["roster_id"] == "content_v234_old" and not item["missing_note"]]
                current = [item for item in method_rows if item["roster_id"] == "content_v6_current" and not item["missing_note"]]
                stat = stratified_exact(([float(item["texture_value"]) for item in old], [float(item["texture_value"]) for item in current]), ([float(item[f"{branch}_margin_{margin_id}"]) for item in old], [float(item[f"{branch}_margin_{margin_id}"]) for item in current])) if len(old) == len(current) == 8 else {"interpretability": "unavailable_incomplete_fixed_denominator"}
                combined = old + current
                row = _association(source=sources[method], branch=branch, margin_id=margin_id, scope="roster_stratified", roster_id="content_v234_old+content_v6_current", fixed_n=16, subset=method_rows, available=combined, stat=stat)
                if roster_stats["content_v234_old"].get("interpretability") == roster_stats["content_v6_current"].get("interpretability") == "available":
                    ro, rc = roster_stats["content_v234_old"]["rho"], roster_stats["content_v6_current"]["rho"]
                    row.update({"rho_old": format(ro, ".17g"), "rho_current": format(rc, ".17g"), "rho_difference": format(rc - ro, ".17g"), "same_nonzero_sign": ro != 0.0 and rc != 0.0 and ((ro > 0) == (rc > 0))})
                result.append(row)
    return result


def _association(*, source: Mapping[str, Any], branch: str, margin_id: str, scope: str, roster_id: str, fixed_n: int, subset: Sequence[dict[str, Any]], available: Sequence[dict[str, Any]], stat: Mapping[str, Any]) -> dict[str, Any]:
    row = {name: "" for name in ASSOCIATION_COLUMNS}
    row.update({"method_id": source["method_id"], "source_exact": source["exact"], "lf_score_domain": source["lf_score_domain"], "branch_role": "primary" if branch == "lf" else "control", "branch": branch, "margin_id": margin_id, "scope": scope, "roster_id": roster_id, "fixed_n": fixed_n, "observed_pair_count": len(available), "statistic_id": "spearman_exact_n8" if scope == "within_roster" else "roster_stratified_spearman_convolution", "interpretability": stat["interpretability"], "missing_unit_ids": ";".join(item["unit_id"] for item in subset if item not in available), "note": "exploratory_descriptive_only"})
    if stat["interpretability"] == "available":
        row.update({"statistic_value": format(stat["rho"], ".17g"), "statistic_be_hex": stat["rho_be_hex"], "c_numerator": stat["c"]["numerator"], "c_denominator": stat["c"]["denominator"], "permutation_scheme": "all_8_factorial_labeled" if scope == "within_roster" else "independent_8_factorial_by_8_factorial_convolution", "permutation_extreme_count": stat["permutation_extreme_count"], "permutation_total_count": stat["permutation_total_count"], "permutation_p_value": format(stat["permutation_p_value"], ".17g"), "permutation_p_be_hex": stat["permutation_p_be_hex"], "texture_median": format(median(float(item["texture_value"]) for item in available), ".17g"), "margin_median": format(median(float(item[f"{branch}_margin_{margin_id}"]) for item in available), ".17g")})
    return row


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows({name: row.get(name, "") for name in columns} for row in rows)
    return stream.getvalue().encode("utf-8")


def _public_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = {name: row[name] for name in ("global_ordinal", "roster_id", "roster_ordinal", "unit_id", "source_id", "seed", "method_id", "source_exact", "lf_score_domain", "status", "failure_class", "plain_ppm_sha256", "plain_rgb_sha256", "texture_be_hex", "texture_rank_be_hex", "candidate_rgb_sha256", "primary_null_rgb_sha256", "primary_null_matches_plain", "joint_or_identity", "reuse_source_method", "missing_note")}
        for field in ("lf_registered", "lf_max_wrong", "lf_null_registered", "lf_margin_a", "lf_margin_b", "hf_registered", "hf_max_wrong", "hf_null_registered", "hf_margin_a", "hf_margin_b"):
            item[f"{field}_be_hex"] = f64_hex(float(row[field])) if row[field] != "" else ""
        result.append(item)
    return result


def _publish_terminal(run_root: Path, run_id: str, members: Sequence[tuple[str, bytes]], staging_parent: Path) -> str:
    with tempfile.TemporaryDirectory(prefix=".texture-terminal-", dir=staging_parent) as staging:
        temporary = Path(staging) / f"{run_id}.zip"
        with temporary.open("xb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, payload in members:
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, payload)
        digest = sha256_bytes(temporary.read_bytes())
        created: list[Path] = []
        run_root_created = False
        terminal_created = False
        terminal = run_root / "terminal"
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            run_root_created = True
            terminal.mkdir()
            terminal_created = True
            archive_path = terminal / temporary.name
            sidecar_path = terminal / f"{temporary.name}.sha256"
            with temporary.open("rb") as source, archive_path.open("xb") as target:
                created.append(archive_path)
                shutil.copyfileobj(source, target)
            with sidecar_path.open("xb") as handle:
                created.append(sidecar_path)
                handle.write(f"{digest}  {archive_path.name}\n".encode("ascii"))
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            if terminal_created:
                terminal.rmdir()
            if run_root_created:
                run_root.rmdir()
            raise
        return digest


def _complete_checkpoint_members(checkpoints: Path) -> list[tuple[str, bytes]]:
    result = []
    for index in range(1, 10):
        path = checkpoints / f"checkpoint-{index:04d}.json"
        sidecar = checkpoints / f"checkpoint-{index:04d}.json.sha256"
        if not path.exists() and not sidecar.exists():
            break
        if not path.is_file() or not sidecar.is_file():
            break
        payload, binding = path.read_bytes(), sidecar.read_bytes()
        if binding != f"{sha256_bytes(payload)}  {path.name}\n".encode("ascii"):
            break
        result.extend(((f"checkpoints/{path.name}", payload), (f"checkpoints/{sidecar.name}", binding)))
    return result


def _public_audit_members(output: Path) -> list[tuple[str, bytes]]:
    result = []
    bindings_path = output / "model_bindings.json"
    if bindings_path.is_file():
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        if isinstance(bindings, dict):
            bindings.pop("hf_home", None)
            result.append(("audit/model_bindings.json", stable_json_bytes(bindings)))
    plain_path = output / "plain_bindings.json"
    if plain_path.is_file():
        plains = json.loads(plain_path.read_text(encoding="utf-8"))
        if isinstance(plains, list):
            result.append(("audit/plain_bindings.json", stable_json_bytes(plains)))
            for plain in plains:
                relative = plain.get("relative_path") if isinstance(plain, dict) and plain.get("status") == "success" else None
                if isinstance(relative, str) and relative.startswith("plain_rgb/") and not Path(relative).is_absolute() and ".." not in Path(relative).parts:
                    path = output / relative
                    if path.is_file():
                        result.append((relative, path.read_bytes()))
    return result


def _publish_operational_terminal(context: Mapping[str, Any], error: Exception) -> str:
    protocol = context["protocol"]
    failure_class = type(error).__name__ if type(error).__name__ in PUBLIC_FAILURES else "OtherOperationalError"
    receipt = {
        "artifact_kind": "operational_terminal",
        "analysis_id": protocol.config["analysis_id"],
        "exact": context["exact"],
        "protocol_digest": protocol.protocol_digest,
        "run_id": context["run_id"],
        "status": "operational_failure",
        "claim_ceiling": protocol.config["claim_ceiling"],
        "failure_class": failure_class,
        "failure_stage": _failure_stage,
        "last_completed_checkpoint": context["last_completed_checkpoint"],
        "checkpoint_scope": "local_transient",
        "resume_allowed": False,
        "result_member": "failure.json",
        "external_validation_required": True,
    }
    result = {key: receipt[key] for key in ("analysis_id", "exact", "protocol_digest", "run_id", "status", "claim_ceiling", "failure_class", "failure_stage", "last_completed_checkpoint", "checkpoint_scope", "resume_allowed")}
    members = [("receipt.json", stable_json_bytes(receipt)), ("failure.json", stable_json_bytes(result))]
    checkpoints = context.get("checkpoints")
    if isinstance(checkpoints, Path):
        members.extend(_complete_checkpoint_members(checkpoints))
    output = context.get("output")
    if isinstance(output, Path):
        members.extend(_public_audit_members(output))
    return _publish_terminal(context["run_root"], context["run_id"], members, context["staging_parent"])


def _execute(args: argparse.Namespace) -> int:
    global _failure_context
    repo = Path(args.repo_root).resolve()
    exact = args.expected_exact
    _set_failure_stage("identity")
    _identity(repo, exact)
    _set_failure_stage("protocol")
    protocol = load_protocol(repo)
    local_parent = Path(args.local_work_root).resolve()
    artifact_sink = Path(args.artifact_sink).resolve()
    provenance_root = Path(args.provenance_root).resolve()
    if local_parent.exists():
        raise FileExistsError("initial-only local work root exists")
    _set_failure_stage("secrets")
    key_text, token = os.environ.pop(KEY_ENV, ""), os.environ.pop(TOKEN_ENV, "")
    if not key_text.strip() or not token.strip():
        key_text = token = ""
        raise RuntimeError("both user secrets are required")
    from cegwm.shared.keys import normalize_detection_key, public_key_digest
    key_digest = public_key_digest(normalize_detection_key(key_text))
    if key_digest != protocol.config["public_key_digest"]:
        key_text = token = ""
        raise RuntimeError("public key digest differs")
    run_id = f"{protocol.run_id_prefix}-{key_digest[:12]}"
    run_root = artifact_sink / exact / run_id
    if run_root.exists():
        key_text = token = ""
        raise FileExistsError("create-only artifact run exists")
    local = local_parent
    _failure_context = {"protocol": protocol, "exact": exact, "run_id": run_id, "run_root": run_root, "staging_parent": local_parent.parent, "checkpoints": None, "output": None, "last_completed_checkpoint": 0}
    local.mkdir(parents=True)
    checkpoints = local / "checkpoints"
    checkpoints.mkdir()
    output = local / "output"
    output.mkdir()
    _failure_context.update({"checkpoints": checkpoints, "output": output})
    _set_failure_stage("checkouts")
    checkouts = _create_checkouts(repo, local / "source_checkouts", protocol.config["sources"])
    old_spec, current_spec = protocol.config["rosters_in_order"]
    _set_failure_stage("rosters")
    units = _load_roster(checkouts["v2"] / old_spec["path"], old_spec["sha256"], old_spec["roster_id"], 0) + _load_roster(checkouts["v6"] / current_spec["path"], current_spec["sha256"], current_spec["roster_id"], 8)
    units_path = local / "units-private.json"
    _write_json_exclusive(units_path, units)
    _set_failure_stage("assets")
    asset_v7 = _stage_asset(provenance_root, local / "public_assets" / "v7", protocol.config["assets"]["v7_iss"])
    asset_v8 = _stage_asset(provenance_root, local / "public_assets" / "v8", protocol.config["assets"]["v8_iss"])
    adapter = repo / "experiments" / "content_texture_stratification_v1_adapter.py"
    cache = local / "hf_home"
    child_env = {KEY_ENV: key_text, TOKEN_ENV: token}
    _set_failure_stage("prefetch")
    prefetch = _child(adapter, checkouts["v2"], protocol.config["sources"]["v2"]["exact"], "asset_prefetch", units_path, output, cache, None, child_env)
    bindings_path = output / "model_bindings.json"
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings["v4_blobs"] = _v4_blob_bindings(checkouts["v4"], checkouts["v5"])
    bindings["source_bindings"] = protocol.config["sources"]
    bindings_path.write_bytes(stable_json_bytes(bindings))
    identity = {"analysis_id": protocol.config["analysis_id"], "exact": exact, "protocol_id": protocol.config["protocol_id"], "protocol_digest": protocol.protocol_digest, "run_id": run_id, "public_key_digest": key_digest, "fixed_plain_units": 16, "fixed_method_rows": 112, "resume_allowed": False}
    _set_failure_stage("common_plain")
    plain_events = _unit_events(_child(adapter, checkouts["v2"], protocol.config["sources"]["v2"]["exact"], "common_plain_v2", units_path, output, cache, bindings_path, child_env), "common_plain", 16)
    plains = []
    state: dict[str, Any] = {"phase": "common_plain", "plain_bindings": [], "method_records": []}
    checkpoint_index = 0
    for event in plain_events:
        if event["status"] == "success":
            event["absolute_path"] = str(output / event["relative_path"])
            texture = parse_p6_texture(Path(event["absolute_path"]).read_bytes())
            event["texture_value"] = format(texture, ".17g")
            event["texture_be_hex"] = f64_hex(texture)
        plains.append(event)
        state["plain_bindings"].append({key: value for key, value in event.items() if key != "absolute_path"})
    checkpoint_index += 1
    _checkpoint(checkpoints, checkpoint_index, identity, state)
    _failure_context["last_completed_checkpoint"] = checkpoint_index
    _write_json_exclusive(output / "plain_bindings.json", state["plain_bindings"])
    method_events: dict[str, list[dict[str, Any]]] = {}
    for method in ("v2", "v3", "v4"):
        _set_failure_stage(method)
        events = _unit_events(_child(adapter, checkouts[method], protocol.config["sources"][method]["exact"], method, units_path, output, cache, bindings_path, child_env), method, 16)
        method_events[method] = events
        state["phase"] = method
        for event in events:
            state["method_records"].append(_checkpoint_event(event))
        checkpoint_index += 1
        _checkpoint(checkpoints, checkpoint_index, identity, state)
        _failure_context["last_completed_checkpoint"] = checkpoint_index
    _set_failure_stage("v5_validate")
    _child(adapter, checkouts["v5"], protocol.config["sources"]["v5"]["exact"], "v5_validate", units_path, output, cache, bindings_path, child_env)
    method_events["v5"] = [{**item, "method": "v5"} for item in method_events["v4"]]
    state["phase"] = "v5_derived"
    for event in method_events["v5"]:
        state["method_records"].append(_checkpoint_event(event))
    checkpoint_index += 1
    _checkpoint(checkpoints, checkpoint_index, identity, state)
    _failure_context["last_completed_checkpoint"] = checkpoint_index
    for method, asset in (("v6", None), ("v7", asset_v7), ("v8", asset_v8)):
        _set_failure_stage(method)
        events = _unit_events(_child(adapter, checkouts[method], protocol.config["sources"][method]["exact"], method, units_path, output, cache, bindings_path, child_env, v7_asset=asset if method == "v7" else None, v8_asset=asset if method == "v8" else None), method, 16)
        method_events[method] = events
        state["phase"] = method
        for event in events:
            state["method_records"].append(_checkpoint_event(event))
        checkpoint_index += 1
        _checkpoint(checkpoints, checkpoint_index, identity, state)
        _failure_context["last_completed_checkpoint"] = checkpoint_index
    key_text = token = ""
    _set_failure_stage("analysis")
    rows = []
    for method in METHOD_ORDER:
        source = protocol.config["sources"][method]
        for unit, plain, event in zip(units, plains, method_events[method]):
            rows.append(_derive_row(unit, method, source, plain, event))
    _rank_rows(rows)
    associations = _association_rows(rows, protocol.config["sources"])
    distinct = all(len({row["texture_be_hex"] for row in rows if row["roster_id"] == roster and row["method_id"] == protocol.config["sources"]["v2"]["method_id"]}) >= 2 for roster in ("content_v234_old", "content_v6_current"))
    complete = len(rows) == 112 and len(associations) == 84 and all(not row["missing_note"] and row["primary_null_matches_plain"] for row in rows) and distinct
    status = "analysis_complete" if complete else "not_interpretable"
    state["phase"] = "analysis"
    state["analysis_status"] = status
    checkpoint_index += 1
    _checkpoint(checkpoints, checkpoint_index, identity, state)
    _failure_context["last_completed_checkpoint"] = checkpoint_index
    if checkpoint_index != 9 or state["phase"] != protocol.config["execution"]["checkpoint_stages"][-1]:
        raise RuntimeError("local transient checkpoint stage count differs")
    _set_failure_stage("terminal_publication")
    per_unit_csv = _csv_bytes(PER_UNIT_COLUMNS, rows)
    associations_csv = _csv_bytes(ASSOCIATION_COLUMNS, associations)
    public_records = _public_records(rows)
    result = {"analysis_id": protocol.config["analysis_id"], "status": status, "claim_ceiling": protocol.config["claim_ceiling"], "fixed_plain_units": 16, "fixed_method_rows": 112, "association_rows": 84, "method_order": list(METHOD_ORDER), "roster_order": [item["roster_id"] for item in protocol.config["rosters_in_order"]], "gate_or_scientific_status_mutated": False, "interpretation": "exploratory_descriptive_only"}
    bindings_public = {"identity": identity, "sources": protocol.config["sources"], "rosters": protocol.config["rosters_in_order"], "assets": protocol.config["assets"], "execution": protocol.config["execution"]}
    if isinstance(bindings.get("manifest_sha256"), str):
        bindings_public["model_manifest_sha256"] = bindings["manifest_sha256"]
    environment = dict(bindings.get("environment_record", {"python": sys.version.split()[0], "platform": sys.platform, "record_only": True}))
    receipt = {"artifact_kind": "terminal", "run_id": run_id, "exact": exact, "protocol_digest": protocol.protocol_digest, "status": status, "result_member": "result.json", "external_validation_required": True}
    members = [("receipt.json", stable_json_bytes(receipt)), ("bindings.json", stable_json_bytes(bindings_public)), ("environment_record.json", stable_json_bytes(environment)), ("records.json", stable_json_bytes(public_records)), ("result.json", stable_json_bytes(result)), ("per_unit.csv", per_unit_csv), ("associations.csv", associations_csv)]
    members.extend((event["relative_path"], Path(event["absolute_path"]).read_bytes()) for event in plains if event["status"] == "success")
    terminal_sha = _publish_terminal(run_root, run_id, members, local)
    print(f"{RESULT_PREFIX} " + json.dumps({"status": status, "claim_ceiling": protocol.config["claim_ceiling"], "exact": exact, "protocol_digest": protocol.protocol_digest, "run_id": run_id, "terminal_sha256": terminal_sha}, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if status == "analysis_complete" else 2


def execute(args: argparse.Namespace) -> int:
    global _failure_context
    _failure_context = None
    try:
        return _execute(args)
    except Exception as error:
        if _failure_context is None:
            raise
        terminal_sha = _publish_operational_terminal(_failure_context, error)
        protocol = _failure_context["protocol"]
        receipt = {
            "artifact_kind": "operational_terminal",
            "status": "operational_failure",
            "claim_ceiling": protocol.config["claim_ceiling"],
            "exact": _failure_context["exact"],
            "protocol_digest": protocol.protocol_digest,
            "run_id": _failure_context["run_id"],
            "terminal_sha256": terminal_sha,
            "failure_class": type(error).__name__ if type(error).__name__ in PUBLIC_FAILURES else "OtherOperationalError",
            "failure_stage": _failure_stage,
            "last_completed_checkpoint": _failure_context["last_completed_checkpoint"],
            "result_member": "failure.json",
        }
        payload = {name: receipt[name] for name in OPERATIONAL_RESULT_FIELDS}
        print(f"{RESULT_PREFIX} " + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        return 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    parser.add_argument("--provenance-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(execute(_arguments()))
    except Exception as error:
        print(_failure_line(error), flush=True)
        raise SystemExit(2) from None
