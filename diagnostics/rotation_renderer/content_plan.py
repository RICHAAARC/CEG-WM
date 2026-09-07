"""Prepared content diagnostic; importing or printing the plan runs no models."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

from PIL import Image

from cegwm.geometry_v7.r1a import condition_by_id
from cegwm.geometry_v7.r1b import rectify_attacked_rgb
from cegwm.runtime.blind_detection import BlindProductionAssets, _score_current_rgb
from cegwm.shared.keys import normalize_detection_key, public_key_digest
from diagnostics.rotation_renderer.run import CONDITIONS, render

UNITS = ("content-v6-iss-eval-0001", "content-v6-iss-eval-0002")
ARMS = ("CG", "G")
FILLS = ("black", "reflect")
FORMAL_TAU_REFERENCE = 1.2657276026437319
ARM_NAMES = {"CG": "CG_with_content_with_sync", "G": "G_no_content_with_sync"}
PILOT = tuple((UNITS[0], arm, CONDITIONS[1], "black") for arm in ARMS)
ALL_ATTACKS = tuple((u, a, c, f) for u in UNITS for a in ARMS for c in CONDITIONS for f in FILLS)
REMAINING = tuple(x for x in ALL_ATTACKS if x not in PILOT)


def plan():
    return dict(status="PREPARED_NOT_EXECUTED", science_denominator=0,
        selection="first two historical units in source order; no result selection",
        units=UNITS, positive="CG", paired_content_negative="G (sync present, content absent)",
        interpolation="bilinear", fills=FILLS, angles=[-15,15],
        attacked_images=16, score_routes=["pre", "predicted_h_post", "oracle"],
        unattacked_reference_images=4, total_content_calls=52,
        pilot=PILOT, pilot_content_calls=6, remaining_content_calls=46,
        additional_geometry_calls=8, cg_geometry="reuse renderer-v1 raw H",
        candidate_protocol="black fill", stress_protocol="reflect fill",
        primary_tau_reference=FORMAL_TAU_REFERENCE,
        threshold_role="historical formal threshold, descriptive only; no recalibration",
        claim="current V1 content statistic diagnostic; not old paired R0 decision or formal blind run")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prepare_sources(inputs):
    """Read artifacts only; require explicit per-arm download mapping."""
    inputs = Path(inputs)
    r0 = _read(inputs / "r0-result.json")
    mapping = _read(inputs / "content_source_mapping.json")
    originals = [x for x in r0["raw_unit_records"] if x["stage"] == "evaluation" and x["unit_id"] in UNITS]
    if tuple(x["unit_id"] for x in originals) != UNITS:
        raise ValueError("requires first two original evaluation records in order")
    sources = {}
    for row in originals:
        if row["residual_strength_multiplier"] != .75:
            raise ValueError("requires original .75 images")
        for arm in ARMS:
            original = [x for x in row["arms"] if x["arm"] == ARM_NAMES[arm]]
            local = [x for x in mapping["images"] if (x["unit_id"],x["arm"]) == (row["unit_id"],arm)]
            if len(original) != 1 or len(local) != 1 or original[0].get("errors"):
                raise ValueError("original arm missing, failed or ambiguous")
            item = local[0]
            if item["original_path"] != original[0]["image_file"]:
                raise ValueError("download mapping differs from original R0 arm")
            path = (inputs / item["local_filename"]).resolve()
            if not path.is_relative_to(inputs.resolve()):
                raise ValueError("local input escaped input directory")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (512,512) or image.format != "PNG":
                    raise ValueError("requires original RGB PNG")
                image.load()
            sources[row["unit_id"],arm] = dict(item, path=str(path))
    return r0, sources


def _score_routes(attacked, predicted_h, truth_h, score, tau):
    """Independent routes from attacked RGB; score sees only the current image."""
    rows = []
    for route in ("pre", "predicted_h_post", "oracle"):
        row = dict(route=route, statistic=None, above_reference_tau=None, error=None, scorer_called=False)
        start = time.perf_counter()
        try:
            if route == "pre":
                current = attacked
            else:
                h = predicted_h if route == "predicted_h_post" else truth_h
                if h is None:
                    raise ValueError("predicted H unavailable")
                current = rectify_attacked_rgb(attacked, h)
            row["scorer_called"] = True
            statistic = score(current)
            row.update(statistic=asdict(statistic), above_reference_tau=statistic.value > tau)
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        row["seconds"] = time.perf_counter()-start
        rows.append(row)
    return rows


def validate_scoring_assets(assets, key, r0):
    """Check existing assets for reuse without constructing or running models."""
    from experiments.run_blind_detection_v1 import (
        load_weighted_asset_semantic, load_whitening_asset_semantic, load_iss_asset_semantic,
    )
    if type(assets) is not BlindProductionAssets:
        raise TypeError("requires current typed production scoring assets")
    root = Path(__file__).resolve().parents[2]
    normalized_key = normalize_detection_key(key)
    if r0["public_key_digest"] != public_key_digest(normalized_key):
        raise ValueError("original embedding and current scoring key differ")
    iss = assets.content_assets.iss_assets
    comparisons = (
        (assets.weighted_joint_asset, load_weighted_asset_semantic(root)),
        (iss.lf_public_assets.whitening_asset, load_whitening_asset_semantic(root)),
        (iss.iss_asset, load_iss_asset_semantic(root)),
    )
    if any(a.payload != b.payload for a,b in comparisons):
        raise ValueError("current production scoring asset semantics differ")
    return normalized_key


class ContentSession:
    """Call only after execution approval, with the existing production assets.

    This class never constructs a model, changes a scorer, or generates an image.
    Production factory construction, if needed, is a separately approved action.
    """
    def __init__(self, inputs, geometry_results, output, key, assets):
        r0, self.sources = prepare_sources(inputs)
        self.key = validate_scoring_assets(assets, key, r0)
        key_identity = public_key_digest(self.key)
        self.assets, self.tau = assets, FORMAL_TAU_REFERENCE
        geometry_path = Path(geometry_results)
        geometry_rows = _read(geometry_path)["rows"] if geometry_path.is_file() else [
            json.loads(line) for name in ("baseline.jsonl","remaining.jsonl")
            for line in (geometry_path/name).read_text().splitlines()]
        self.cg_geometry = {}
        for row in geometry_rows:
            if row["unit_id"] in UNITS and row["interpolation"] == "bilinear":
                key = (row["unit_id"],row["condition_id"],row["fill"])
                if key in self.cg_geometry:
                    raise ValueError("duplicate saved CG geometry")
                if row["original_image_file"] != self.sources[row["unit_id"],"CG"]["original_path"]:
                    raise ValueError("saved CG geometry belongs to another image")
                self.cg_geometry[key] = row["geometry"]
        if set(self.cg_geometry) != {(u,c,f) for u in UNITS for c in CONDITIONS for f in FILLS}:
            raise ValueError("saved CG geometry incomplete")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = False
        self.score = lambda image: _score_current_rgb(image, self.key, self.assets)
        metadata = dict(plan(), status="CONTENT_DIAGNOSTIC_SESSION_CREATED",
            tau_reference=self.tau, key_identity=key_identity,
            loaded_asset_tau_record_only=assets.threshold_asset.tau_blind if assets.threshold_asset else None,
            r0_exact=r0.get("exact"), current_scoring="cegwm.runtime.blind_detection._score_current_rgb",
            current_asset_paths=["configs/content_chain/assets", "configs/blind_detection/assets/blind_detection_v1_thresholds.json"],
            sources=list(self.sources.values()))
        (self.output/"plan.json").write_text(json.dumps(metadata,indent=2))

    def _attack(self, case):
        unit, arm, condition, fill = case
        source = self.sources[unit,arm]
        try:
            with Image.open(source["path"]) as image:
                attacked = render(image, condition, fill, "bilinear")
        except Exception as exc:
            return dict(case=case, source=source, geometry=None,
                scores=[dict(route=r, statistic=None, above_reference_tau=None,
                    scorer_called=False,
                    error="render:"+type(exc).__name__+": "+str(exc))
                    for r in ("pre","predicted_h_post","oracle")])
        error = None
        try:
            geometry = self.cg_geometry[unit,condition,fill] if arm == "CG" else asdict(self.assets.geometry_backend.detect_geometry(attacked))
            h = geometry["homography_observed_to_canonical"]
            if not geometry["legal"] or geometry["error"] is not None:
                h = None
        except Exception as exc:
            geometry, h, error = None, None, type(exc).__name__ + ": " + str(exc)
        truth = condition_by_id(condition).truth_observed_to_canonical
        rows = _score_routes(attacked,h,truth,self.score,self.tau)
        return dict(case=case, source=source, geometry=geometry, geometry_error=error,
            geometry_source="saved CG" if arm == "CG" else "current G detection", scores=rows)

    def pilot(self):
        if self.started:
            raise ValueError("pilot cannot be repeated")
        self.started = True
        start = time.perf_counter()
        results = []
        with (self.output/"pilot.jsonl").open("x") as file:
            for case in PILOT:
                row = self._attack(case)
                results.append(row)
                file.write(json.dumps(row) + "\n")
                file.flush()
        elapsed = time.perf_counter()-start
        return dict(pilot_seconds=elapsed, planned_content_calls=6,
            completed_statistics=sum(x["statistic"] is not None for row in results for x in row["scores"]),
            actual_score_calls=sum(x["scorer_called"] for row in results for x in row["scores"]),
            route_errors=sum(x["error"] is not None for row in results for x in row["scores"]),
            remaining_calls=46, rough_remaining_seconds=elapsed*46/6,
            estimate_note="initialization excluded; different routes can cost differently; inspect failures first")

    def remaining(self):
        if not self.started:
            raise ValueError("run and inspect pilot first")
        with (self.output/"remaining.jsonl").open("x") as file:
            for unit in UNITS:
                for arm in ARMS:
                    row = dict(unit=unit, arm=arm, route="unattacked", statistic=None, error=None, scorer_called=False)
                    try:
                        with Image.open(self.sources[unit,arm]["path"]) as image:
                            row["scorer_called"] = True
                            statistic = self.score(image)
                        row.update(statistic=asdict(statistic),above_reference_tau=statistic.value>self.tau)
                    except Exception as error:
                        row["error"] = type(error).__name__ + ": " + str(error)
                    file.write(json.dumps(row)+"\n")
                    file.flush()
            for case in REMAINING:
                file.write(json.dumps(self._attack(case))+"\n")
                file.flush()
        self._write_table()

    def _write_table(self):
        rows = []
        for name in ("pilot.jsonl","remaining.jsonl"):
            for line in (self.output/name).read_text().splitlines():
                record = json.loads(line)
                if "case" in record:
                    u,a,c,f = record["case"]
                    scores = record["scores"]
                else:
                    u,a,c,f = record["unit"],record["arm"],"identity","none"
                    scores = [record]
                for row in scores:
                    rows.append(dict(unit=u,arm=a,condition=c,fill=f,route=row["route"],
                        m=row["statistic"]["value"] if row["statistic"] else None,
                        above_reference_tau=row.get("above_reference_tau"),
                        scorer_called=row["scorer_called"],error=row["error"]))
        with (self.output/"paired.csv").open("x",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(planned_routes=52, recorded_routes=len(rows),
            actual_score_calls=sum(x["scorer_called"] for x in rows),
            complete_statistics=sum(x["m"] is not None for x in rows),
            route_errors=sum(x["error"] is not None for x in rows), science_denominator=0)
        (self.output/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2))
