"""Runnable training, checkpoint selection and fixed-denominator RGB8 evaluation."""
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .adapter import pil_image, rgb8
from .config import ARMS, SHIFTS
from .data import file_hash, json_write
from .execution import legacy_profile, configure_torch
from .networks import Reader, Writer, parameter_counts
from .protocol import (bootstrap_sources, digest, interior, key_bits, reference,
                       reference_target, schedule, shift_rgb, stage_a,
                       summarize_prediction, wilson)


def tensor_signature(items):
    return hashlib.sha256(repr(items).encode()).hexdigest()


def identity(cache, config):
    return {"manifest_hash": cache.hash, "config_hash": config.digest}


def rng_state():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), *s[2:]))
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA resume requires original CUDA environment")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def load_checkpoint(path, cache, config, arm, device):
    import json
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file() or json.loads(sidecar.read_text()).get("sha256") != file_hash(path):
        raise ValueError("checkpoint checksum missing or mismatched")
    state = torch.load(path, map_location=device, weights_only=True)
    for k, v in {**identity(cache, config), "arm": arm}.items():
        if state.get(k) != v:
            raise ValueError("checkpoint identity mismatch: " + k)
    if cache.raw.get("execution_hash") is not None and state.get("execution_hash") != cache.raw["execution_hash"]:
        raise ValueError("checkpoint execution profile differs from frozen manifest")
    if type(state.get("update")) is not int or not 1 <= state["update"] <= config.updates:
        raise ValueError("checkpoint update outside frozen budget")
    if state.get("torch_version") != str(torch.__version__):
        raise ValueError("checkpoint torch version mismatch; exact resume requires pinned environment")
    expected = schedule(config.seed, state["update"], len(cache.split("train")), config.batch_size)
    if state["next_input_signature"] != tensor_signature(expected):
        raise ValueError("resume input schedule mismatch")
    return state


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def check_clean(adapter, z, clean):
    with torch.no_grad():
        expected = rgb8(adapter.decode(z))
    if not torch.equal(expected.cpu(), clean.cpu()):
        raise ValueError("cached clean RGB8 is not the decode of this final scaled z0/VAE")


def training_batch(cache, rows, config, step, device):
    chosen = schedule(config.seed, step, len(rows), config.batch_size)
    zs, cleans, bits, ps, attacks = [], [], [], [], []
    for index, attack, nonce in chosen:
        z, clean = cache.load(rows[index])
        zs.append(z)
        cleans.append(clean)
        bits.append(key_bits(rows[index]["key"]))
        ps.append(reference(nonce, config.latent_size))
        attacks.append(SHIFTS[attack])
    return (torch.stack(zs).to(device), torch.stack(cleans).to(device),
            torch.stack(bits).to(device), torch.stack(ps).to(device), attacks, chosen)


def train(cache, config, adapter, arm, output, *, device="cpu", resume=None, stop_after=None, execution=None, progress=None):
    """stop_after is an interruption boundary, never changes frozen update budget."""
    if torch.device(device).type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for deterministic CUDA training")
    execution = execution or legacy_profile(config)
    execution.validate(config)
    if cache.raw.get("execution_hash") not in (None, execution.digest):
        raise ValueError("training execution profile differs from frozen manifest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = cache.split("train")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    configure_torch(device)
    writer, reader = Writer(arm, config).to(device), Reader(arm, config).to(device)
    optimizer = torch.optim.AdamW([*writer.parameters(), *reader.parameters()],
                                 lr=config.lr, weight_decay=config.weight_decay)
    start, elapsed_prior = 0, 0.0
    if resume:
        state = load_checkpoint(resume, cache, config, arm, device)
        if state.get("execution_hash", legacy_profile(config).digest) != execution.digest:
            raise ValueError("resume execution profile mismatch")
        writer.load_state_dict(state["writer"])
        reader.load_state_dict(state["reader"])
        optimizer.load_state_dict(state["optimizer"])
        if state["device"] != str(device):
            raise ValueError("exact resume requires the original device")
        restore_rng(state["rng"])
        start, elapsed_prior = state["update"], state["elapsed_seconds"]
    elif list(output.glob("*.pt")):
        raise ValueError("existing checkpoints require explicit --resume")
    end = config.updates if stop_after is None else min(config.updates, stop_after)
    if not start < end:
        raise ValueError("resume update must precede stop boundary")
    checked = set()
    began = time.perf_counter()
    last_path = None
    for step in range(start, end):
        z, clean, bits, p, attacks, chosen = training_batch(cache, rows, config, step, device)
        for slot, (index, _, _) in enumerate(chosen):
            source = rows[index]["source_id"]
            if source not in checked:
                check_clean(adapter, z[slot:slot+1], clean[slot:slot+1])
                checked.add(source)
        optimizer.zero_grad(set_to_none=True)
        losses = dict.fromkeys(("total", "bit", "reference", "rgb", "residual"), 0.0)
        from torch.utils.checkpoint import checkpoint
        for offset in range(0, config.batch_size, execution.microbatch):
            sl = slice(offset, offset+execution.microbatch)
            zm, bm, pm = z[sl], bits[sl], p[sl]
            zw = writer(zm, bm, None if arm == "B0" else pm)
            decoded = checkpoint(adapter.decode, zw, use_reentrant=False) if execution.checkpoint_vae else adapter.decode(zw)
            quantized = adapter.quantize_ste(decoded)
            attacked = torch.cat([shift_rgb(quantized[i:i+1], *a) for i, a in enumerate(attacks[sl])])
            observation = checkpoint(adapter.encode_proxy, attacked, use_reentrant=False) if execution.checkpoint_vae else adapter.encode_proxy(attacked)
            read = reader(observation)
            evidence = read["evidence"]
            bit_loss = (1-bm*evidence).mean() if arm == "C" else F.binary_cross_entropy_with_logits(evidence, (bm+1)/2)
            reference_loss = zm.new_zeros(())
            if arm != "B0":
                targets = torch.cat([reference_target(pm[i:i+1], *a, config.downsample) for i,a in enumerate(attacks[sl])])
                reference_loss = F.mse_loss(interior(read["p"], config.mask_border), interior(targets, config.mask_border))
            rgb_loss = F.mse_loss(quantized, clean[sl].float()/255)
            residual_loss = F.mse_loss(zw, zm)
            loss = bit_loss + config.reference_weight*reference_loss + config.rgb_weight*rgb_loss + config.residual_weight*residual_loss
            if not bool(torch.isfinite(loss)):
                json_write(output/"failure.json", {"update": step, "status": "engineering_error", "reason": "nonfinite_loss"})
                raise ValueError("nonfinite training loss")
            weight = execution.microbatch/config.batch_size
            (loss*weight).backward()
            for name, value in zip(losses, (loss, bit_loss, reference_loss, rgb_loss, residual_loss)):
                losses[name] += float(value.detach())*weight
            # Do not retain the last microbatch's graph across optimizer updates.
            del zw, decoded, quantized, attacked, observation, read, evidence, loss
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in [*writer.parameters(), *reader.parameters()]):
            raise ValueError("nonfinite training gradient")
        optimizer.step()
        update = step + 1
        if progress:
            progress({"arm": arm, "update": update, "budget": config.updates, "loss": losses})
        if update % config.checkpoint_every == 0 or update % execution.recovery_every == 0 or update == end:
            from dataclasses import asdict
            state = {**identity(cache, config), "arm": arm, "update": update,
                     "execution_hash": execution.digest, "execution": asdict(execution),
                     "writer": writer.state_dict(), "reader": reader.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state(),
                     "elapsed_seconds": elapsed_prior + time.perf_counter()-began,
                     "next_input_signature": tensor_signature(schedule(config.seed, update, len(rows), config.batch_size)),
                     "last_input_signature": tensor_signature(chosen), "torch_version": str(torch.__version__),
                     "device": str(device), "loss": losses}
            prefix = "update" if update % config.checkpoint_every == 0 or execution.name == "batch-v1" else "recovery"
            last_path = output / f"{prefix}_{update:06d}.pt"
            if last_path.exists():
                raise ValueError("refusing to overwrite checkpoint")
            temporary = last_path.with_suffix(".tmp")
            with open(temporary, "wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(last_path)
            json_write(last_path.with_suffix(".json"), {k: state[k] for k in (
                "manifest_hash", "config_hash", "arm", "update", "elapsed_seconds", "last_input_signature",
                "next_input_signature", "loss", "torch_version", "device", "execution_hash")} | {"sha256": file_hash(last_path)})
    return last_path


def networks_from_checkpoint(path, cache, config, arm, device):
    state = load_checkpoint(path, cache, config, arm, device)
    writer, reader = Writer(arm, config).to(device), Reader(arm, config).to(device)
    writer.load_state_dict(state["writer"])
    reader.load_state_dict(state["reader"])
    return writer.eval(), reader.eval(), state


@torch.no_grad()
def measure(cache, config, adapter, arm, checkpoint, split, shifts, *, device="cpu", metric=None, samples=None):
    writer, reader, state = networks_from_checkpoint(checkpoint, cache, config, arm, device)
    rows, records = cache.split(split), []
    synchronize(device)
    began = time.perf_counter()
    detection_seconds = 0.0
    for index, row in enumerate(rows):
        common = {"unit_id": row["unit_id"], "source": row["source_id"], "source_id": row["source_id"],
                  "source_group": row["source_group"], "arm": arm}
        try:
            z, clean = cache.load(row)
            z, clean = z[None].to(device), clean[None].to(device)
            check_clean(adapter, z, clean)
            p = reference(row["reference_nonce"], config.latent_size)[None].to(device)
            bits = key_bits(row["key"])[None].to(device)
            marked = rgb8(adapter.decode(writer(z, bits, None if arm == "B0" else p)))
            mse = float(((marked.float()-clean.float())/255).square().mean())
            psnr = -10 * math.log10(mse) if mse else None
            perceptual, quality_error = None, None
            if metric is not None:
                try:
                    perceptual = float(metric(marked.float()/255, clean.float()/255)[0])
                    if not math.isfinite(perceptual) or perceptual < 0:
                        raise ValueError("invalid LPIPS measurement")
                except Exception as error:
                    perceptual = None
                    quality_error = f"{type(error).__name__}: {error}"
            if samples is not None and index < 4:
                directory = Path(samples)
                directory.mkdir(parents=True, exist_ok=True)
                pil_image(marked[0]).save(directory / f"{index:03d}_{arm}_marked.png")
                pil_image(clean[0]).save(directory / f"{index:03d}_clean.png")
            for dx, dy in shifts:
                record = {**common, "shift": [dx, dy], "psnr": psnr, "rgb_exact": mse == 0, "lpips": perceptual,
                          "quality_error": quality_error}
                try:
                    attacked = shift_rgb(marked, dx, dy)
                    synchronize(device)
                    detection_start = time.perf_counter()
                    read = reader(adapter.observe(attacked))
                    evidence = read["evidence"][0]
                    if not bool(torch.isfinite(evidence).all()):
                        raise ValueError("nonfinite reader evidence")
                    energy = float(interior(read["p"], config.mask_border).float().square().sum()) if read["p"] is not None else None
                    record.update(summarize_prediction(evidence, row["key"]))
                    synchronize(device)
                    duration = time.perf_counter()-detection_start
                    detection_seconds += duration
                    record["detection_seconds"] = duration
                    record.update({"reference_energy": energy,
                                   "status": "low_reference_energy" if energy is not None and energy <= config.epsilon else (
                                       "no_evidence" if not bool(evidence.abs().sum()) else "ok")})
                except Exception as error:
                    record.update({"status": "engineering_error", "error": f"{type(error).__name__}: {error}",
                                   "top1": False, "bit_accuracy": 0.0, "exact": False})
                records.append(record)
                if os.environ.get("CEGWM_PROGRESS") == "1":
                    print(f"measure arm={arm} split={split} unit={index+1}/{len(rows)} shift=({dx},{dy}) status={record['status']}", flush=True)
        except Exception as error:
            for dx, dy in shifts:
                records.append({**common, "shift": [dx, dy], "status": "engineering_error",
                                "error": f"{type(error).__name__}: {error}", "top1": False,
                                "bit_accuracy": 0.0, "exact": False, "psnr": None, "lpips": None})
            if os.environ.get("CEGWM_PROGRESS") == "1":
                print(f"measure arm={arm} split={split} unit={index+1}/{len(rows)} engineering_error={type(error).__name__}", flush=True)
    synchronize(device)
    return {"records": records, "total_evaluation_seconds": time.perf_counter()-began,
            "detection_seconds": detection_seconds,
            "timing_scope": "uint8 image -> existing E -> reader -> K17 scores (includes PIL/CPU transfers); excludes writer/shift/quality",
            "device": str(device), "hardware": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else __import__("platform").processor(),
            "torch_threads": torch.get_num_threads(),
            "training_seconds": state["elapsed_seconds"], "update": state["update"],
            "checkpoint_sha256": file_hash(checkpoint)}


def summarize(records):
    n = len(records)
    wins = sum(r["top1"] for r in records)
    return {"denominator": n, "top1_count": wins, "top1_rate": wins/n if n else 0,
            "top1_wilson95": wilson(wins, n), "bit_accuracy": sum(r["bit_accuracy"] for r in records)/n if n else 0,
            "whole_message_exact_count": sum(r["exact"] for r in records),
            "status_counts": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
            "complete": n > 0 and all(r["status"] != "engineering_error" for r in records),
            "ci_note": "Wilson interval treats units as independent; source-cluster bootstrap supplied for B"}


def select_checkpoints(cache, config, adapter, run_dir, output, *, device="cpu"):
    """Only model_selection data. Tie order: top1, bit accuracy, earliest update."""
    selected, audits = {}, {}
    for arm in ARMS:
        options = sorted((Path(run_dir)/arm).glob("update_*.pt"))
        if not options:
            raise ValueError("no checkpoints for " + arm)
        audits[arm] = []
        expected_updates = set(range(config.checkpoint_every, config.updates+1, config.checkpoint_every))
        found_updates = []
        eligible_paths = []
        for path in options:
            state = load_checkpoint(path, cache, config, arm, device)
            if state["update"] % config.checkpoint_every or state["update"] > config.updates:
                continue  # Interrupted snapshots are resumable, not additional selection chances.
            found_updates.append(state["update"])
            eligible_paths.append(path)
        if set(found_updates) != expected_updates or len(found_updates) != len(expected_updates) or config.updates not in found_updates:
            raise ValueError("all arms require complete budget and identical frozen checkpoint opportunities: " + arm)
        for path in eligible_paths:
            result = measure(cache, config, adapter, arm, path, "model_selection", [SHIFTS[0]], device=device)
            summary = summarize(result["records"])
            audits[arm].append({"path": str(path.resolve()), "sha256": file_hash(path), "update": result["update"], **summary})
        eligible = [r for r in audits[arm] if r["complete"]]
        if not eligible:
            raise ValueError("no complete model_selection checkpoint for " + arm)
        best = max(eligible, key=lambda r: (r["top1_rate"], r["bit_accuracy"], -r["update"]))
        selected[arm] = {k: best[k] for k in ("path", "sha256", "update")}
    report = {**identity(cache, config), "selection_split": "model_selection", "selected": selected, "audits": audits}
    if Path(output).exists():
        raise ValueError("selection lock already exists")
    json_write(output, report)
    return report


def validate_selection(selection, cache, config):
    import json
    result = json.loads(Path(selection).read_text())
    if result.get("selection_split") != "model_selection" or any(result.get(k) != v for k, v in identity(cache, config).items()):
        raise ValueError("selection identity mismatch")
    if set(result["selected"]) != set(ARMS):
        raise ValueError("all three arms required")
    for row in result["selected"].values():
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("selected weights changed")
    return result


def quality_report(by_arm, config):
    report = {"status": "not_evaluable", "arms": {}, "paired": {},
              "rule": "all units; PSNR>=30dB, LPIPS<=.05; C-minus-each baseline PSNR>=-.5dB, LPIPS<=.005"}
    complete = True
    arrays = {}
    for arm, records in by_arm.items():
        records = sorted(records, key=lambda r: r["unit_id"])
        # Exact RGB equality has infinite PSNR. Keep it explicit; do not cap/drop it.
        psnr = [r.get("psnr") for r in records]
        lpips = [r.get("lpips") for r in records]
        if any(x is None or not math.isfinite(x) for x in psnr+lpips):
            complete = False
            report["arms"][arm] = {"status": "not_evaluable", "denominator": len(records),
                                    "exact_rgb_count": sum(r.get("rgb_exact", False) for r in records),
                                    "reason": "missing measurement or infinite PSNR; no finite matched difference"}
            continue
        p, l = np.asarray(psnr), np.asarray(lpips)
        arrays[arm] = (p, l)
        report["arms"][arm] = {"denominator": len(records), "mean_psnr": float(p.mean()),
                                "p05_psnr": float(np.quantile(p, .05)), "mean_lpips": float(l.mean()),
                                "p95_lpips": float(np.quantile(l, .95)), "psnr": psnr, "lpips": lpips}
    if not complete:
        return report
    passed = all(p.mean() >= config.psnr_min and l.mean() <= config.lpips_max for p, l in arrays.values())
    for arm in ("B0", "B1"):
        dp, dl = arrays["C"][0]-arrays[arm][0], arrays["C"][1]-arrays[arm][1]
        report["paired"][arm] = {"mean_psnr_difference": float(dp.mean()), "mean_lpips_difference": float(dl.mean()),
                                  "psnr_differences": dp.tolist(), "lpips_differences": dl.tolist()}
        passed = passed and dp.mean() >= config.psnr_gap_min and dl.mean() <= config.lpips_gap_max
    report["status"] = "pass_pilot_budget" if passed else "fail_pilot_budget"
    return report


def stage_b_report(by_arm, config):
    sources = sorted({r["source"] for r in by_arm["C"]})
    groups = {r["source"]: r.get("source_group", r["source"]) for r in by_arm["C"]}
    clusters = sorted(set(groups.values()))
    def cluster_ci(values):
        rng = np.random.default_rng(config.seed)
        blocks = [np.asarray([i for i,s in enumerate(sources) if groups[s] == group]) for group in clusters]
        means = [float(values[np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])].mean())
                 for _ in range(config.bootstrap_samples)]
        return np.quantile(means, [.025, .975]).tolist()
    vectors, result = {}, {"source_denominator": len(sources), "cluster_denominator": len(clusters), "arms": {}, "paired": {}}
    for arm, records in by_arm.items():
        vectors[arm] = np.asarray([all(r["top1"] for r in records if r["source"] == source) for source in sources], dtype=float)
        vector = vectors[arm]
        result["arms"][arm] = {"all_four_source_rate": float(vector.mean()), "all_four_source_count": int(vector.sum()),
            "source_bootstrap95": cluster_ci(vector),
            "directions": {str(s): summarize([r for r in records if r["shift"] == list(s)]) for s in SHIFTS[1:]}}
    gains = []
    for arm in ("B0", "B1"):
        diff = vectors["C"]-vectors[arm]
        gains.append(float(diff.mean()))
        result["paired"][arm] = {"gain": float(diff.mean()), "rescue_sources": int((diff > 0).sum()),
            "harm_sources": int((diff < 0).sum()), "paired_source_bootstrap95": cluster_ci(diff)}
    result["complete"] = all(r["status"] != "engineering_error" for records in by_arm.values() for r in records)
    result["gain_target_met"] = all(g >= config.b_gain_target for g in gains) and result["complete"]
    result["interpretation"] = "Image-level all-keys/four-shifts success; bootstrap resamples source_group clusters; 10pp is a pilot target, not proof"
    return result


def evaluate(cache, config, adapter, selection, split, output, *, device="cpu", metric=None, dev_report=None, metric_info=None):
    import json
    if split not in ("dev", "test"):
        raise ValueError("evaluation uses dev or locked test; calibration is reserved")
    if Path(output).exists():
        raise ValueError("refusing to overwrite frozen evaluation")
    lock = validate_selection(selection, cache, config)
    selection_hash = file_hash(selection)
    metric_info = metric_info or {"status": "injected_test_metric" if metric else "unavailable", "sha256": None}
    permission = None
    if split == "test":
        if dev_report is None:
            raise ValueError("test needs a frozen dev A report")
        dev = json.loads(Path(dev_report).read_text())
        if dev.get("split") != "dev" or dev.get("selection_hash") != selection_hash or any(dev.get(k) != v for k,v in identity(cache, config).items()):
            raise ValueError("dev gate identity mismatch")
        if dev.get("lpips", {}).get("sha256") != metric_info.get("sha256"):
            raise ValueError("test quality metric differs from frozen dev metric")
        permission = bool(dev["A_gate"]["pass"])
    zero, costs = {}, {}
    for arm in ARMS:
        result = measure(cache, config, adapter, arm, lock["selected"][arm]["path"], split, [SHIFTS[0]],
                         device=device, metric=metric, samples=str(output)+".samples")
        zero[arm], costs[arm] = result.pop("records"), result
    summaries = {arm: summarize(rows) for arm, rows in zero.items()}
    gate = stage_a({a: s["top1_rate"] for a,s in summaries.items()}, all(s["complete"] for s in summaries.values()), config)
    allow_b = gate["pass"] if split == "dev" else permission
    report = {**identity(cache, config), "selection_hash": selection_hash, "split": split,
              "A": summaries, "A_gate": gate if split == "dev" else {"status": "descriptive_only_test_not_for_decisions", "dev_pass": permission},
              "A_records": zero, "quality": quality_report(zero, config), "costs": costs,
              "lpips": metric_info,
              "parameters": parameter_counts(config), "B": {"status": "skipped_by_dev_A_gate"},
              "research_status": "not_evaluable", "scientific_claim": "unvalidated pilot; no TAR/FPR or arbitrary geometry claim"}
    if allow_b:
        shifted = {}
        for arm in ARMS:
            result = measure(cache, config, adapter, arm, lock["selected"][arm]["path"], split, SHIFTS[1:], device=device)
            shifted[arm] = result["records"]
            report["costs"][arm]["B_total_evaluation_seconds"] = result["total_evaluation_seconds"]
            report["costs"][arm]["B_detection_seconds"] = result["detection_seconds"]
        report["B"] = stage_b_report(shifted, config)
        report["B_records"] = shifted
        if not report["B"]["complete"] or not all(s["complete"] for s in summaries.values()):
            report["research_status"] = "engineering_error"
        elif report["quality"]["status"] == "fail_pilot_budget":
            report["research_status"] = "current_configuration_quality_budget_failed"
        elif report["quality"]["status"] == "pass_pilot_budget":
            report["research_status"] = "continue_pilot" if report["B"]["gain_target_met"] else "current_configuration_no_gain"
    elif split == "dev":
        report["research_status"] = gate["status"]
    json_write(output, report)
    return report


@torch.no_grad()
def four_write_diagnostic(cache, config, adapter, selection, output, *, device="cpu"):
    """Fixed first DEV host, first two DEV keys, two independently supplied DEV nonces."""
    lock = validate_selection(selection, cache, config)
    rows, keys = cache.split("dev"), cache.keys["dev"]
    if len(keys) < 2 or len(rows) < 2:
        raise ValueError("diagnostic requires two dev keys and two dev nonce rows")
    row = rows[0]
    z, clean = cache.load(row)
    z, clean = z[None].to(device), clean[None].to(device)
    check_clean(adapter, z, clean)
    records = []
    for arm in ARMS:
        writer, reader, _ = networks_from_checkpoint(lock["selected"][arm]["path"], cache, config, arm, device)
        for pi in range(2):
            p = reference(rows[pi]["reference_nonce"], config.latent_size)[None].to(device)
            for bi in range(2):
                bits = key_bits(keys[bi])[None].to(device)
                marked = rgb8(adapter.decode(writer(z, bits, None if arm == "B0" else p)))
                result = reader(adapter.observe(marked))
                record = {"arm": arm, "reference_index": pi, "message_index": bi,
                          **summarize_prediction(result["evidence"][0], keys[bi])}
                if result["p"] is not None:
                    record["reference_mse"] = float(F.mse_loss(interior(result["p"], config.mask_border), interior(p, config.mask_border)))
                records.append(record)
                directory = Path(str(output)+".samples")
                directory.mkdir(parents=True, exist_ok=True)
                pil_image(marked[0]).save(directory / f"{arm}_p{pi}_b{bi}.png")
    report = {**identity(cache, config), "source": row["source_id"], "records": records,
              "limitation": "diagnostic cannot exclude ideal implicit classification then field synthesis"}
    json_write(output, report)
    return report
