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
    domain_margins,
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
    "prefetch", "common_plain", "c2", "c3", "c6", "v2", "v3", "v4", "v5_validate", "v6", "v7", "v8", "analysis", "terminal_publication",
})
_failure_stage = "identity"
# C2/C3/C6 are the only executed candidate constructions.  V4 is a C3
# whitened-LF rescore and V5 is a C3 alias, not extra candidate/scorer rows.
METHOD_ORDER = ("c2", "c3", "c6")
CONSTRUCTION_SOURCE = {"c2": "v2", "c3": "v3", "c6": "v6"}
DOMAIN_MATRIX = {"c2": ("ordinary_lf", "hf"), "c3": ("ordinary_lf", "v4_lf", "hf"), "c6": ("v4_lf", "hf")}
SCORE_FIELDS = tuple(f"{branch}__{label}" for branch in ("lf", "hf", "joint") for label in ("registered", *(f"wrong_{index:02d}" for index in range(16))))
OPERATIONAL_RESULT_FIELDS = ("artifact_kind", "status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256", "failure_class", "failure_stage", "last_completed_checkpoint", "result_member")
_failure_context: dict[str, Any] | None = None
PER_UNIT_COLUMNS = (
    "global_ordinal", "roster_id", "roster_ordinal", "unit_id", "source_id", "seed", "method_id", "source_exact", "lf_score_domain", "status", "failure_class", "plain_ppm_sha256", "plain_rgb_sha256", "texture_value", "texture_be_hex", "texture_rank", "texture_rank_be_hex", "candidate_rgb_sha256", "primary_null_rgb_sha256", "primary_null_matches_plain", "ordinary_lf_registered", "ordinary_lf_max_wrong", "ordinary_lf_null_registered", "ordinary_lf_margin_a", "ordinary_lf_margin_b", "v4_lf_registered", "v4_lf_max_wrong", "v4_lf_null_registered", "v4_lf_margin_a", "v4_lf_margin_b", "hf_registered", "hf_max_wrong", "hf_null_registered", "hf_margin_a", "hf_margin_b", "joint_or_identity", "reuse_source_method", "missing_note",
)
ASSOCIATION_COLUMNS = (
    "method_id", "source_exact", "lf_score_domain", "branch_role", "branch", "margin_id", "scope", "roster_id", "fixed_n", "observed_pair_count", "statistic_id", "statistic_value", "statistic_be_hex", "c_numerator", "c_denominator", "permutation_scheme", "permutation_extreme_count", "permutation_total_count", "permutation_p_value", "permutation_p_be_hex", "texture_median", "margin_median", "rho_old", "rho_current", "rho_difference", "same_nonzero_sign", "interpretability", "missing_unit_ids", "note",
)


def _set_failure_stage(stage: str) -> None:
    global _failure_stage
    if stage not in FAILURE_STAGES:
        raise ValueError("texture failure stage differs")
    _failure_stage = stage


def _null_cache_put(cache: dict[tuple[str, int, str], Mapping[str, Any]], domain: str, ordinal: int, plain_sha: str, scores: Mapping[str, Any]) -> None:
    key = (domain, ordinal, plain_sha)
    if key in cache and cache[key] != scores:
        raise ValueError("domain null cache conflict")
    cache[key] = dict(scores)


def _retain_unit_failure(event: dict[str, Any], *causes: Mapping[str, Any] | None, fallback: str = "RuntimeError") -> None:
    """Keep an attributable unit in the fixed denominator without a traceback."""
    for cause in causes:
        failure = cause.get("failure_class") if isinstance(cause, Mapping) else None
        if failure in PUBLIC_FAILURES:
            event["status"] = "operational_failure"
            event["failure_class"] = failure
            return
    event["status"] = "operational_failure"
    event["failure_class"] = fallback


def _join_domain_maps(construction: str, candidate: Mapping[str, Mapping[str, Any]], cache: Mapping[tuple[str, int, str], Mapping[str, Any]], ordinal: int, plain_sha: str) -> dict[str, tuple[float, float]]:
    from cegwm.protocol.content_texture_stratification_v1 import require_construction_domains
    domains = DOMAIN_MATRIX[construction]
    nulls = {}
    for domain in domains:
        value = cache.get((domain, ordinal, plain_sha))
        if value is None:
            raise ValueError("required domain null missing")
        nulls[domain] = value
    return require_construction_domains(construction, candidate, nulls)


def _required_association_matrix() -> tuple[tuple[str, str, str], ...]:
    return tuple((construction, domain, margin) for construction, domains in DOMAIN_MATRIX.items() for domain in domains for margin in ("a", "b"))


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


def _load_roster(path: Path, expected_sha: str, roster_id: str, offset: int = 0) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise RuntimeError("roster bytes differ")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 96:
        raise ValueError("texture roster must contain exactly 96 units")
    units = []
    for ordinal, row in enumerate(rows, 1):
        if set(row) != {"unit_id", "block_id", "slot_index", "semantic_family", "source_id", "prompt", "seed", "height", "width"} or row["height"] != 512 or row["width"] != 512:
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
    result = {name: value for name, value in event.items() if name not in {"candidate_scores", "null_scores", "candidate_ppm_sha256", "candidate_raw_rgb_sha256"}}
    for group in ("candidate_scores", "null_scores"):
        if group in event:
            result[f"{group}_be_hex"] = {domain: {name: f64_hex(float(score)) for name, score in values.items()} for domain, values in event[group].items()}
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
    if event.get("event") == "unit":
        if event.get("status") == "operational_failure":
            if set(event) != {"event", "method", "global_ordinal", "unit_id", "status", "failure_class"} or event["method"] not in METHOD_ORDER or event["failure_class"] not in PUBLIC_FAILURES:
                raise ValueError("adapter unit failure fields differ")
            return event
        required = {"event", "method", "global_ordinal", "unit_id", "status", "candidate_rgb_sha256", "primary_null_rgb_sha256", "candidate_scores", "null_scores"}
        if set(event) not in (required, required | {"candidate_ppm_sha256", "candidate_raw_rgb_sha256"}):
            raise ValueError("adapter unit event fields differ")
        construction = event["method"]
        domains = ("ordinary_lf", "hf") if construction == "c3" and "candidate_ppm_sha256" in event else DOMAIN_MATRIX.get(construction)
        null_domains = () if construction in {"c3", "c6"} else domains
        if domains is None or set(event["candidate_scores"]) != set(domains) or set(event["null_scores"]) != set(null_domains):
            raise ValueError("adapter scorer domains differ")
        event["candidate_scores"] = {domain: event["candidate_scores"][domain] for domain in domains}
        event["null_scores"] = {domain: event["null_scores"][domain] for domain in null_domains}
        from cegwm.protocol.content_texture_stratification_v1 import require_domain_scores, require_construction_domains
        if (construction == "c3" and "candidate_ppm_sha256" in event) or construction == "c6":
            for domain in domains:
                require_domain_scores(event["candidate_scores"][domain], domain)
                if domain in event["null_scores"]:
                    require_domain_scores(event["null_scores"][domain], domain)
        else:
            require_construction_domains(construction, event["candidate_scores"], event["null_scores"])
    elif event.get("event") == "v4_lf_rescore":
        if event.get("status") == "operational_failure":
            if set(event) != {"event", "method", "global_ordinal", "unit_id", "status", "failure_class"} or event["method"] != "c3" or event["failure_class"] not in PUBLIC_FAILURES:
                raise ValueError("adapter V4 LF failure fields differ")
            return event
        required = {"event", "method", "global_ordinal", "unit_id", "status", "candidate_rgb_sha256", "plain_rgb_sha256", "candidate_ppm_sha256", "plain_ppm_sha256", "candidate_scores", "null_scores"}
        if set(event) != required or event["method"] != "c3" or set(event["candidate_scores"]) != {"v4_lf"} or set(event["null_scores"]) != {"v4_lf"}:
            raise ValueError("adapter V4 LF event fields differ")
        from cegwm.protocol.content_texture_stratification_v1 import require_domain_scores
        require_domain_scores(event["candidate_scores"]["v4_lf"], "v4_lf")
        require_domain_scores(event["null_scores"]["v4_lf"], "v4_lf")
    return event


def _child(adapter: Path, source: Path, exact: str, phase: str, units_path: Path, output: Path, cache: Path, bindings_path: Path | None, env: Mapping[str, str], *, v7_asset: Path | None = None, v8_asset: Path | None = None, transient_root: Path | None = None, transient_bindings: Path | None = None) -> list[dict[str, Any]]:
    command = [sys.executable, str(adapter), "--source-root", str(source), "--expected-exact", exact, "--phase", phase, "--units-json", str(units_path), "--plain-bindings-json", str(output / "plain_bindings.json"), "--local-output-root", str(output), "--hf-cache-root", str(cache)]
    if bindings_path is not None:
        command += ["--model-bindings-json", str(bindings_path)]
    if v7_asset is not None:
        command += ["--v7-asset-root", str(v7_asset)]
    if v8_asset is not None:
        command += ["--v8-asset-root", str(v8_asset)]
    if transient_root is not None:
        command += ["--transient-root", str(transient_root)]
    if transient_bindings is not None:
        command += ["--transient-bindings-json", str(transient_bindings)]
    child_env = {**os.environ, **env, "HF_HOME": str(cache)}
    process = subprocess.Popen(command, cwd=source, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="strict")
    expected_events = 3 if phase == "asset_prefetch" else 98
    events = []
    assert process.stdout is not None
    for line in process.stdout:
        if len(line) > 65536:
            process.kill()
            raise RuntimeError("adapter output line exceeds bound")
        if line.startswith(EVENT_PREFIX):
            events.append(_adapter_event(line))
            if len(events) > expected_events:
                process.kill()
                raise RuntimeError("adapter event count exceeds bound")
    rc = process.wait()
    if rc != 0 or len(events) != expected_events or events[-1] != {"event": "phase_complete", "phase": phase}:
        raise RuntimeError("adapter phase failed or ended out of order")
    return events


def _unit_events(events: Sequence[Mapping[str, Any]], method: str, expected: int) -> list[dict[str, Any]]:
    event_name = "plain" if method == "common_plain" else ("v4_lf_rescore" if method == "c3_v4_lf_rescore" else "unit")
    selected = [dict(item) for item in events if item.get("event") == event_name]
    if len(selected) != expected or [item["global_ordinal"] for item in selected] != list(range(1, expected + 1)):
        raise RuntimeError("adapter unit event order differs")
    if method not in {"common_plain", "c3_v4_lf_rescore"} and any(item.get("method") != method for item in selected):
        raise RuntimeError("adapter method event identity differs")
    return selected


def _derive_row(unit: Mapping[str, Any], method: str, source: Mapping[str, Any], plain: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    base = {name: "" for name in PER_UNIT_COLUMNS}
    base.update({"global_ordinal": unit["global_ordinal"], "roster_id": unit["roster_id"], "roster_ordinal": unit["roster_ordinal"], "unit_id": unit["unit_id"], "source_id": unit["source_id"], "seed": unit["seed"], "method_id": method, "source_exact": source["exact"], "lf_score_domain": source["lf_score_domain"], "status": event["status"], "failure_class": event.get("failure_class", ""), "plain_ppm_sha256": plain.get("plain_ppm_sha256", ""), "plain_rgb_sha256": plain.get("plain_rgb_sha256", ""), "joint_or_identity": "joint_min_recorded_not_analyzed", "reuse_source_method": "v3_for_v4_v5" if method == "c3" else ""})
    if event["status"] != "success" or plain.get("status") != "success":
        base["missing_note"] = "retained_unit_or_plain_failure"
        return base
    scores, null_scores = event["candidate_scores"], event["null_scores"]
    texture = float(plain["texture_value"])
    primary_null_sha = event["primary_null_rgb_sha256"]
    matches_plain = primary_null_sha == plain["plain_rgb_sha256"]
    base.update({"texture_value": format(texture, ".17g"), "texture_be_hex": f64_hex(texture), "candidate_rgb_sha256": event["candidate_rgb_sha256"], "primary_null_rgb_sha256": primary_null_sha, "primary_null_matches_plain": matches_plain})
    for domain in DOMAIN_MATRIX[method]:
        candidate, null = scores[domain], null_scores[domain]
        margin_a, margin_b = domain_margins(candidate, null, domain)
        base.update({
            f"{domain}_registered": format(float(candidate["registered"]), ".17g"),
            f"{domain}_max_wrong": format(max(float(candidate[f"wrong_{index:02d}"]) for index in range(16)), ".17g"),
            f"{domain}_null_registered": format(float(null["registered"]), ".17g"),
            f"{domain}_margin_a": format(margin_a, ".17g"),
            f"{domain}_margin_b": format(margin_b, ".17g"),
        })
    if not matches_plain:
        base["missing_note"] = "method_primary_null_differs_from_dedicated_plain"
    return base


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    from cegwm.protocol.content_texture_stratification_v1 import average_ranks
    plain_by_unit: dict[tuple[str, str], tuple[str, str]] = {}
    for block in range(12):
        subset = rows_by_method(rows)[METHOD_ORDER[0]][block * 8:(block + 1) * 8]
        if len(subset) != 8 or any(not row["texture_value"] for row in subset):
            continue
        ranks = average_ranks([float(row["texture_value"]) for row in subset])
        for row, rank in zip(subset, ranks):
            plain_by_unit[(row["roster_id"], row["unit_id"])] = (str(float(rank)), f64_hex(float(rank)))
    for row in rows:
        rank = plain_by_unit.get((row["roster_id"], row["unit_id"]))
        if rank:
            row["texture_rank"], row["texture_rank_be_hex"] = rank


def rows_by_method(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in METHOD_ORDER}
    for name in METHOD_ORDER:
        result[name] = rows[name_index(name) * 96:(name_index(name) + 1) * 96]
    return result


def name_index(name: str) -> int:
    return METHOD_ORDER.index(name)


def _n96_spearman(texture: Sequence[float], response: Sequence[float]) -> dict[str, Any]:
    """Tie-aware midrank Spearman for the fixed, non-replaceable N=96."""
    from cegwm.protocol.content_texture_stratification_v1 import average_ranks
    if len(texture) != len(response) != 96 or any(not math.isfinite(float(value)) for value in (*texture, *response)):
        return {"interpretability": "unavailable_incomplete_fixed_denominator"}
    x, y = average_ranks(texture), average_ranks(response)
    xm = sum(float(value) for value in x) / 96.0; ym = sum(float(value) for value in y) / 96.0
    numerator = sum((float(left) - xm) * (float(right) - ym) for left, right in zip(x, y))
    xss = sum((float(value) - xm) ** 2 for value in x); yss = sum((float(value) - ym) ** 2 for value in y)
    if xss <= 0.0 or yss <= 0.0:
        return {"interpretability": "unavailable_zero_rank_variance"}
    rho = numerator / math.sqrt(xss * yss)
    if not math.isfinite(rho):
        return {"interpretability": "unavailable_nonfinite"}
    return {"interpretability": "available", "rho": rho, "rho_be_hex": f64_hex(rho)}


def _association_rows(rows: list[dict[str, Any]], sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    grouped = rows_by_method(rows)
    # The prospective analysis has one fixed N=96 roster.  Keep the 12 blocks
    # as the permutation strata; branch/scorer rows never add observations.
    for method, domain, margin_id in _required_association_matrix():
                subset = grouped[method]
                available = [item for item in subset if not item["missing_note"] and item[f"{domain}_margin_{margin_id}"]]
                stat = _n96_spearman([float(item["texture_value"]) for item in available], [float(item[f"{domain}_margin_{margin_id}"]) for item in available]) if len(available) == 96 else {"interpretability": "unavailable_incomplete_fixed_denominator"}
                result.append(_association(source=sources[CONSTRUCTION_SOURCE[method]], branch=domain, margin_id=margin_id, scope="fixed_n96_12x8_block_preserving", roster_id="content_texture_n96_evaluation_v1", fixed_n=96, subset=subset, available=available, stat=stat))
    return result

    # Historical two-roster code is unreachable and retained only pending a
    # later mechanical deletion outside this narrowly scoped correction.
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
                    result.append(_association(source=sources[CONSTRUCTION_SOURCE[method]], branch=branch, margin_id=margin_id, scope="within_roster", roster_id=roster, fixed_n=8, subset=subset, available=available, stat=stat))
                old = [item for item in method_rows if item["roster_id"] == "content_v234_old" and not item["missing_note"]]
                current = [item for item in method_rows if item["roster_id"] == "content_v6_current" and not item["missing_note"]]
                stat = stratified_exact(([float(item["texture_value"]) for item in old], [float(item["texture_value"]) for item in current]), ([float(item[f"{branch}_margin_{margin_id}"]) for item in old], [float(item[f"{branch}_margin_{margin_id}"]) for item in current])) if len(old) == len(current) == 8 else {"interpretability": "unavailable_incomplete_fixed_denominator"}
                combined = old + current
                row = _association(source=sources[CONSTRUCTION_SOURCE[method]], branch=branch, margin_id=margin_id, scope="roster_stratified", roster_id="content_v234_old+content_v6_current", fixed_n=16, subset=method_rows, available=combined, stat=stat)
                if roster_stats["content_v234_old"].get("interpretability") == roster_stats["content_v6_current"].get("interpretability") == "available":
                    ro, rc = roster_stats["content_v234_old"]["rho"], roster_stats["content_v6_current"]["rho"]
                    row.update({"rho_old": format(ro, ".17g"), "rho_current": format(rc, ".17g"), "rho_difference": format(rc - ro, ".17g"), "same_nonzero_sign": ro != 0.0 and rc != 0.0 and ((ro > 0) == (rc > 0))})
                result.append(row)
    return result


def _association(*, source: Mapping[str, Any], branch: str, margin_id: str, scope: str, roster_id: str, fixed_n: int, subset: Sequence[dict[str, Any]], available: Sequence[dict[str, Any]], stat: Mapping[str, Any]) -> dict[str, Any]:
    row = {name: "" for name in ASSOCIATION_COLUMNS}
    row.update({"method_id": source["method_id"], "source_exact": source["exact"], "lf_score_domain": branch, "branch_role": "primary" if branch != "hf" else "control", "branch": branch, "margin_id": margin_id, "scope": scope, "roster_id": roster_id, "fixed_n": fixed_n, "observed_pair_count": len(available), "statistic_id": "block_preserving_spearman_record_only", "interpretability": stat["interpretability"], "missing_unit_ids": ";".join(item["unit_id"] for item in subset if item not in available), "note": "exploratory_descriptive_only"})
    if stat["interpretability"] == "available":
        row.update({"statistic_value": format(stat["rho"], ".17g"), "statistic_be_hex": stat["rho_be_hex"], "permutation_scheme": "12_block_preserving_record_only", "texture_median": format(median(float(item["texture_value"]) for item in available), ".17g"), "margin_median": format(median(float(item[f"{branch}_margin_{margin_id}"]) for item in available), ".17g")})
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
        for field in ("ordinary_lf_registered", "ordinary_lf_max_wrong", "ordinary_lf_null_registered", "ordinary_lf_margin_a", "ordinary_lf_margin_b", "v4_lf_registered", "v4_lf_max_wrong", "v4_lf_null_registered", "v4_lf_margin_a", "v4_lf_margin_b", "hf_registered", "hf_max_wrong", "hf_null_registered", "hf_margin_a", "hf_margin_b"):
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
    # Only the three production sources needed for the minimal mechanism matrix
    # are checked out.  Historical V4/V5/V7/V8 remain source bindings, not work.
    selected_sources = {name: protocol.config["sources"][name] for name in ("v2", "v3", "v4", "v6")}
    checkouts = _create_checkouts(repo, local / "source_checkouts", selected_sources)
    roster_spec = protocol.config["rosters_in_order"][0]
    _set_failure_stage("rosters")
    units = _load_roster(repo / roster_spec["path"], roster_spec["sha256"], roster_spec["roster_id"])
    units_path = local / "units-private.json"
    _write_json_exclusive(units_path, units)
    _set_failure_stage("assets")
    adapter = repo / "experiments" / "content_texture_stratification_v1_adapter.py"
    cache = local / "hf_home"
    child_env = {KEY_ENV: key_text, TOKEN_ENV: token}
    _set_failure_stage("prefetch")
    prefetch = _child(adapter, checkouts["v2"], protocol.config["sources"]["v2"]["exact"], "asset_prefetch", units_path, output, cache, None, child_env)
    bindings_path = output / "model_bindings.json"
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings["source_bindings"] = protocol.config["sources"]
    bindings_path.write_bytes(stable_json_bytes(bindings))
    identity = {"analysis_id": protocol.config["analysis_id"], "exact": exact, "protocol_id": protocol.config["protocol_id"], "protocol_digest": protocol.protocol_digest, "run_id": run_id, "public_key_digest": key_digest, "fixed_plain_units": 96, "fixed_method_rows": 288, "resume_allowed": False}
    _set_failure_stage("common_plain")
    plain_events = _unit_events(_child(adapter, checkouts["v2"], protocol.config["sources"]["v2"]["exact"], "common_plain_v2", units_path, output, cache, bindings_path, child_env), "common_plain", 96)
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
    # Candidate PPMs exist only in this coordinator-owned directory.  It is
    # removed even if V4 scoring or the later analysis fails.
    with tempfile.TemporaryDirectory(prefix=".texture-c3-", dir=local) as transient_name:
        transient = Path(transient_name)
        for construction in METHOD_ORDER:
            source_name = CONSTRUCTION_SOURCE[construction]
            _set_failure_stage(construction)
            events = _unit_events(_child(adapter, checkouts[source_name], protocol.config["sources"][source_name]["exact"], construction, units_path, output, cache, bindings_path, child_env, transient_root=transient if construction == "c3" else None), construction, 96)
            if construction == "c3":
                private_bindings: dict[str, dict[str, str]] = {}
                for event, plain in zip(events, plains):
                    if event.get("status") != "success" or plain.get("status") != "success":
                        continue
                    ppm = event.get("candidate_ppm_sha256")
                    raw = event.get("candidate_raw_rgb_sha256")
                    relative = f"c3/{event['global_ordinal']:03d}.ppm"
                    if not isinstance(ppm, str) or not isinstance(raw, str) or event["candidate_rgb_sha256"] != raw:
                        raise RuntimeError("C3 transient candidate binding differs")
                    plain_relative = plain.get("relative_path")
                    if not isinstance(plain_relative, str):
                        raise RuntimeError("common plain relative binding differs")
                    source_plain = output / plain_relative
                    local_plain = transient / f"plain/{event['global_ordinal']:03d}.ppm"
                    local_plain.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_plain, local_plain)
                    private_bindings[str(event["global_ordinal"])] = {"candidate_relative": relative, "candidate_ppm_sha256": ppm, "candidate_rgb_sha256": raw, "plain_relative": f"plain/{event['global_ordinal']:03d}.ppm", "plain_ppm_sha256": sha256_bytes(local_plain.read_bytes()), "plain_rgb_sha256": plain["plain_rgb_sha256"]}
                transient_bindings = transient / "c3-bindings.json"
                _write_json_exclusive(transient_bindings, private_bindings)
                v4_events = _unit_events(_child(adapter, checkouts["v4"], protocol.config["sources"]["v4"]["exact"], "c3_v4_lf_rescore", units_path, output, cache, bindings_path, child_env, transient_root=transient, transient_bindings=transient_bindings), "c3_v4_lf_rescore", 96)
                by_ordinal = {event["global_ordinal"]: event for event in v4_events}
                for event, plain in zip(events, plains):
                    v4_event = by_ordinal.get(event["global_ordinal"])
                    if event.get("status") != "success" or v4_event is None or v4_event.get("status") != "success":
                        _retain_unit_failure(event, v4_event, fallback="RuntimeError")
                        continue
                    if event["candidate_rgb_sha256"] != v4_event["candidate_rgb_sha256"] or plain.get("plain_rgb_sha256") != v4_event["plain_rgb_sha256"]:
                        _retain_unit_failure(event, fallback="ValueError")
                        continue
                    event["candidate_scores"] = {"ordinary_lf": event["candidate_scores"]["ordinary_lf"], "v4_lf": v4_event["candidate_scores"]["v4_lf"], "hf": event["candidate_scores"]["hf"]}
                    event["null_scores"] = {"v4_lf": v4_event["null_scores"]["v4_lf"]}
                    event.pop("candidate_ppm_sha256", None)
                    event.pop("candidate_raw_rgb_sha256", None)
            method_events[construction] = events
            state["phase"] = construction
            for event in events:
                state["method_records"].append(_checkpoint_event(event))
            checkpoint_index += 1
            _checkpoint(checkpoints, checkpoint_index, identity, state)
            _failure_context["last_completed_checkpoint"] = checkpoint_index
    # The common-null cache is keyed by scorer domain and immutable common
    # plain identity.  C2 produces ordinary/HF; the detached V4 phase produces
    # V4-LF.  Consumers receive the producer values rather than rescoring.
    null_cache: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for event, plain in zip(method_events["c2"], plains):
        if event.get("status") == "success" and plain.get("status") == "success":
            for domain in ("ordinary_lf", "hf"):
                _null_cache_put(null_cache, domain, event["global_ordinal"], plain["plain_rgb_sha256"], event["null_scores"][domain])
    for event, plain in zip(method_events["c3"], plains):
        if event.get("status") == "success" and plain.get("status") == "success":
            try:
                for domain in ("ordinary_lf", "hf"):
                    event["null_scores"][domain] = dict(null_cache[(domain, event["global_ordinal"], plain["plain_rgb_sha256"])])
                _null_cache_put(null_cache, "v4_lf", event["global_ordinal"], plain["plain_rgb_sha256"], event["null_scores"]["v4_lf"])
                event["null_scores"] = {domain: event["null_scores"][domain] for domain in DOMAIN_MATRIX["c3"]}
                from cegwm.protocol.content_texture_stratification_v1 import require_construction_domains
                require_construction_domains("c3", event["candidate_scores"], event["null_scores"])
            except (KeyError, TypeError, ValueError):
                _retain_unit_failure(event, method_events["c2"][event["global_ordinal"] - 1], fallback="RuntimeError")
    for event, plain in zip(method_events["c6"], plains):
        if event.get("status") == "success" and plain.get("status") == "success":
            try:
                for domain in ("v4_lf", "hf"):
                    event["null_scores"][domain] = dict(null_cache[(domain, event["global_ordinal"], plain["plain_rgb_sha256"])])
            except (KeyError, TypeError, ValueError):
                _retain_unit_failure(event, method_events["c2"][event["global_ordinal"] - 1], method_events["c3"][event["global_ordinal"] - 1], fallback="RuntimeError")
    key_text = token = ""
    _set_failure_stage("analysis")
    rows = []
    for method in METHOD_ORDER:
        source = protocol.config["sources"][CONSTRUCTION_SOURCE[method]]
        for unit, plain, event in zip(units, plains, method_events[method]):
            rows.append(_derive_row(unit, method, source, plain, event))
    _rank_rows(rows)
    associations = _association_rows(rows, protocol.config["sources"])
    distinct = len({row["texture_be_hex"] for row in rows if row["method_id"] == "c2"}) >= 2
    complete = (len(rows) == 288 and all(not row["missing_note"] and row["primary_null_matches_plain"] for row in rows)
                and len(associations) == 14 and all(row["interpretability"] == "available" and row["observed_pair_count"] == 96 and row["statistic_value"] != "" for row in associations)
                and distinct)
    status = "analysis_complete" if complete else "not_interpretable"
    state["phase"] = "analysis"
    state["analysis_status"] = status
    checkpoint_index += 1
    _checkpoint(checkpoints, checkpoint_index, identity, state)
    _failure_context["last_completed_checkpoint"] = checkpoint_index
    _set_failure_stage("terminal_publication")
    per_unit_csv = _csv_bytes(PER_UNIT_COLUMNS, rows)
    associations_csv = _csv_bytes(ASSOCIATION_COLUMNS, associations)
    public_records = _public_records(rows)
    result = {"analysis_id": protocol.config["analysis_id"], "status": status, "claim_ceiling": protocol.config["claim_ceiling"], "fixed_plain_units": 96, "fixed_method_rows": 288, "association_rows": len(associations), "method_order": list(METHOD_ORDER), "roster_order": [item["roster_id"] for item in protocol.config["rosters_in_order"]], "gate_or_scientific_status_mutated": False, "interpretation": "exploratory_descriptive_only"}
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
