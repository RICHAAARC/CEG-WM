"""One isolated Colab stage per process. Heavy imports stay inside stage functions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from .colab_config import GENERATION, MODEL_CARD, file_sha, json_hash, roster
from .colab_storage import atomic_json, completed_asset, quarantine, read_json, redact, write_asset
from .config import Config
from .execution import ExecutionProfile, configure_torch


def announce(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def context(root):
    root = Path(root)
    run = read_json(root/"run.json")
    data, checksum = roster(run["repo"])
    import subprocess
    commit = subprocess.check_output(["git", "-C", run["repo"], "rev-parse", "HEAD"], text=True).strip()
    expected = {"code_commit": commit, "config_hash": Config().digest, "generator_hash": json_hash(GENERATION),
                "sources_hash": checksum, "model_revision": GENERATION["revision"], "execution_hash": ExecutionProfile().digest}
    if run["identity"] != expected:
        raise ValueError("worker code/config/generator/roster/execution identity mismatch")
    return run, data


def hardware(root, run):
    configure_torch("cuda")
    import torch
    import torchvision
    import psutil
    from diffusers import StableDiffusion3Pipeline, AutoencoderKL
    from transformers import T5EncoderModel, CLIPTextModel, CLIPTextModelWithProjection
    import sentencepiece
    import lpips
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no CPU fallback")
    gpu = torch.cuda.get_device_properties(0)
    gpu_gib, ram_gib = gpu.total_memory/2**30, psutil.virtual_memory().total/2**30
    if not torch.cuda.is_bf16_supported():
        raise ValueError("native BF16 GPU required")
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "sentencepiece", "lpips", "numpy",
        "tokenizers", "safetensors", "Pillow", "scipy", "protobuf")}
    if versions["diffusers"] != GENERATION["diffusers"]:
        raise ValueError("pinned diffusers version required")
    runtime = {"versions": versions, "python": sys.version.split()[0], "gpu": gpu.name,
               "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability(0)),
               "execution_hash": ExecutionProfile().digest}
    path = root/"runtime.json"
    if path.exists() and read_json(path)["identity"] != runtime:
        raise ValueError("runtime differs from this run's frozen environment; do not mix/resume")
    capacity = {"gpu_gib": gpu_gib, "ram_gib": ram_gib, "capacity_policy": "report_only",
                "capacity_note": "Capacity is informational, not an admission gate; actual stages may run out of memory."}
    atomic_json(path, {"identity": runtime, **capacity})
    announce(stage="hardware", **runtime, **capacity)


def inventory(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(directory).parts}


def prepare_model(root, run):
    from huggingface_hub import HfApi, snapshot_download, get_hf_file_metadata, hf_hub_url
    directory = Path(run["base"])/"models"/GENERATION["revision"]
    snapshot = directory/"snapshot"
    marker = directory/"prepared.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["model_id"] != GENERATION["model_id"] or saved["revision"] != GENERATION["revision"] or inventory(snapshot) != saved["files"]:
            raise ValueError("prepared model identity/hash conflict")
    else:
        token = os.environ.get("HF_TOKEN")
        try:
            get_hf_file_metadata(hf_hub_url(GENERATION["model_id"], "transformer/config.json", revision=GENERATION["revision"]), token=token)
        except Exception as error:
            # No raw HTTP body/headers in diagnostics.
            raise RuntimeError(f"HF gate/access check failed ({type(error).__name__}). Accept model terms at {MODEL_CARD}, then provide HF_TOKEN in Colab Secrets.") from None
        info = HfApi().model_info(GENERATION["model_id"], revision=GENERATION["revision"], token=token, files_metadata=True)
        if info.sha != GENERATION["revision"]:
            raise ValueError("HF revision mismatch")
        prefixes = ("scheduler/", "tokenizer/", "tokenizer_2/", "tokenizer_3/", "text_encoder/", "text_encoder_2/", "text_encoder_3/", "transformer/", "vae/")
        files = [x.rfilename for x in info.siblings if (x.rfilename == "model_index.json" or x.rfilename.startswith(prefixes)) and ".fp16" not in x.rfilename]
        selected = [x for x in info.siblings if x.rfilename in files]
        if any(x.size is None for x in selected):
            raise ValueError("HF file sizes unavailable; cannot preflight storage")
        directory.mkdir(parents=True, exist_ok=True)
        missing_bytes = sum(x.size for x in selected if not (snapshot/x.rfilename).is_file() or (snapshot/x.rfilename).stat().st_size != x.size)
        required_bytes = missing_bytes + 8*2**30
        available = shutil.disk_usage(directory).free
        atomic_json(root/"storage_preflight.json", {"download_bytes_missing": missing_bytes, "reserve_bytes": 8*2**30,
                    "required_free_bytes": required_bytes, "observed_free_bytes": available,
                    "note": "DriveFS free-space estimate; shared Google account quota may impose additional limits"})
        if available < required_bytes:
            raise ValueError(f"Insufficient Drive space: need {required_bytes/2**30:.1f}GiB free (downloads plus 8GiB reserve), observed {available/2**30:.1f}GiB; no large download started")
        snapshot_download(GENERATION["model_id"], revision=GENERATION["revision"], local_dir=str(snapshot), allow_patterns=files, token=token)
        actual = inventory(snapshot)
        if set(actual) != set(files):
            raise ValueError("model download file inventory differs from frozen HF revision")
        for item in selected:
            if item.lfs is not None:
                official = item.lfs.sha256 if hasattr(item.lfs, "sha256") else item.lfs["sha256"]
                if actual[item.rfilename] != official:
                    raise ValueError("model weight differs from official LFS SHA256: " + item.rfilename)
            else:
                payload = (snapshot/item.rfilename).read_bytes()
                if hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest() != item.blob_id:
                    raise ValueError("model metadata differs from official Git blob: " + item.rfilename)
        saved = {"model_id": GENERATION["model_id"], "revision": GENERATION["revision"], "files": actual,
                 "denoiser_files": {k:v for k,v in actual.items() if k.startswith("transformer/")}}
        saved["denoiser_sha256"] = json_hash(saved["denoiser_files"])
        atomic_json(marker, saved)
    vae = root/"cache"/"vae"
    vae.mkdir(parents=True, exist_ok=True)
    for name, checksum in saved["files"].items():
        if not name.startswith("vae/"):
            continue
        dest = vae/name.removeprefix("vae/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if file_sha(dest) != checksum:
                raise ValueError("local VAE copy hash conflict")
        else:
            temporary = dest.with_suffix(dest.suffix+".tmp")
            shutil.copyfile(snapshot/name, temporary)
            os.replace(temporary, dest)
    atomic_json(root/"model.json", {**saved, "snapshot": str(snapshot), "vae_files": inventory(vae)})


def prepare_lpips(root):
    import torch
    import torchvision
    import lpips
    from urllib.request import urlopen
    directory = root/"quality"
    directory.mkdir(parents=True, exist_ok=True)
    identity = json_hash({k:v for k,v in GENERATION.items() if "alexnet" in k or "lpips" in k})
    completed = completed_asset(directory, "lpips-full.pt", identity)
    if completed:
        provenance = directory/"provenance.json"
        if provenance.exists() and read_json(provenance).get("full_sha256") == completed["sha256"]:
            return
        quarantine([directory/"lpips-full.pt", directory/"lpips-full.pt.json"], directory, "incomplete LPIPS provenance; rebuild from official assets")
    backbone = directory/"alexnet-owt-7be5be79.pth"
    if not backbone.exists():
        temporary = backbone.with_suffix(".tmp")
        torch.hub.download_url_to_file(GENERATION["alexnet_url"], str(temporary), hash_prefix=GENERATION["alexnet_sha256_prefix"], progress=True)
        os.replace(temporary, backbone)
    if not file_sha(backbone).startswith(GENERATION["alexnet_sha256_prefix"]):
        raise ValueError("AlexNet official checksum mismatch")
    linear = directory/"alex-linear.pth"
    if not linear.exists():
        payload = urlopen(GENERATION["lpips_linear_url"]).read()
        blob = hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()
        if blob != GENERATION["lpips_linear_git_blob"]:
            raise ValueError("LPIPS immutable git blob mismatch")
        temporary = linear.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, linear)
    payload = linear.read_bytes()
    if hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest() != GENERATION["lpips_linear_git_blob"]:
        raise ValueError("LPIPS linear file changed")
    alex = torchvision.models.alexnet(weights=None)
    alex.load_state_dict(torch.load(backbone, map_location="cpu", weights_only=True), strict=True)
    metric = lpips.LPIPS(net="alex", pnet_rand=True, pretrained=False, verbose=False)
    for i in range(1, 6):
        for index, layer in getattr(metric.net, f"slice{i}")._modules.items():
            layer.load_state_dict(alex.features[int(index)].state_dict(), strict=True)
    calibrated = torch.load(linear, map_location="cpu", weights_only=True)
    if set(calibrated) != {f"lin{i}.model.1.weight" for i in range(5)}:
        raise ValueError("unexpected LPIPS linear architecture")
    metric.load_state_dict(calibrated, strict=False)
    for key, value in calibrated.items():
        if not torch.equal(metric.state_dict()[key], value):
            raise ValueError("LPIPS calibration did not load")
    write_asset(directory, "lpips-full.pt", identity, lambda p: torch.save(metric.state_dict(), p))
    atomic_json(directory/"provenance.json", {"backbone_url": GENERATION["alexnet_url"], "backbone_sha256": file_sha(backbone),
        "linear_url": GENERATION["lpips_linear_url"], "linear_sha256": file_sha(linear), "linear_git_blob": GENERATION["lpips_linear_git_blob"],
        "full_sha256": file_sha(directory/"lpips-full.pt"), "random_backbone": False})


def source_identity(run, row, root):
    return json_hash({"run": run["identity"], "source": row, "runtime": read_json(root/"runtime.json")["identity"]})


def clean_identity(run, row, root, latent):
    return json_hash({"source": source_identity(run,row,root), "latent_sha256": latent["sha256"]})


def cache_status(root, run, data):
    missing_latent, missing_clean = [], []
    for index, row in enumerate(data["sources"]):
        stem = row["source_id"]
        latent = completed_asset(root/"cache"/"latent", stem+".npy", source_identity(run,row,root))
        if latent is None:
            missing_latent.append(index)
            # Clean pixels without their completed latent cannot be reused as a pair.
            directory = root/"cache"/"clean"
            quarantine([directory/(stem+".png"), directory/(stem+".png.json")], directory, "latent incomplete; keep dependent clean as evidence and regenerate pair")
            missing_clean.append(index)
        elif completed_asset(root/"cache"/"clean", stem+".png", clean_identity(run,row,root,latent)) is None:
            missing_clean.append(index)
    missing_vae = not (root/"model.json").exists()
    if not missing_vae and inventory(root/"cache"/"vae") != read_json(root/"model.json")["vae_files"]:
        raise ValueError("completed local VAE identity/hash conflict")
    if not missing_latent and not missing_clean and not missing_vae and (root/"cache"/"manifest.json").exists():
        from .data import Cache
        Cache(root/"cache"/"manifest.json", Config())
    result = {"missing_latent": missing_latent, "missing_clean": missing_clean, "missing_vae": missing_vae}
    atomic_json(root/"cache_status.json", result)
    announce(stage="cache_status", missing_latent=len(missing_latent), missing_clean=len(missing_clean), missing_vae=missing_vae)


def generate(root, run, rows):
    directory = root/"cache"/"latent"
    missing = [row for row in rows if not completed_asset(directory, row["source_id"]+".npy", source_identity(run,row,root))]
    if not missing:
        announce(stage="generate", reused=len(rows))
        return
    import inspect
    import torch
    configure_torch("cuda")
    from diffusers import StableDiffusion3Pipeline
    import diffusers
    if diffusers.__version__ != GENERATION["diffusers"]:
        raise ValueError("generation requires pinned diffusers")
    source = inspect.getsource(StableDiffusion3Pipeline.__call__)
    if 'if output_type == "latent":\n            image = latents' not in source:
        raise ValueError("pipeline latent return semantics changed")
    model = read_json(root/"model.json")
    pipe = StableDiffusion3Pipeline.from_pretrained(model["snapshot"], torch_dtype=torch.bfloat16, local_files_only=True)
    if any(getattr(pipe, name, None) is None for name in ("text_encoder", "text_encoder_2", "text_encoder_3")):
        raise ValueError("all three text encoders required")
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    for row in missing:
        announce(stage="generate", source=row["source_id"], split=row["split"], status="started")
        with torch.inference_mode():
            latent = pipe(prompt=row["prompt"], height=512, width=512, num_inference_steps=40,
                          guidance_scale=4.5, max_sequence_length=256, output_type="latent",
                          generator=torch.Generator(device="cpu").manual_seed(row["generation_seed"])).images
        if latent.shape != (1, 16, 64, 64) or not bool(torch.isfinite(latent).all()):
            raise ValueError("pipeline did not return finite final scaled z0")
        import numpy as np
        array = latent[0].float().cpu().numpy()
        def save(path):
            with open(path, "wb") as f:
                np.save(f, array, allow_pickle=False)
        write_asset(directory, row["source_id"]+".npy", source_identity(run,row,root), save)
        announce(stage="generate", source=row["source_id"], status="completed")


def clean(root, run, rows):
    directory = root/"cache"/"clean"
    missing = []
    for row in rows:
        latent = completed_asset(root/"cache"/"latent", row["source_id"]+".npy", source_identity(run,row,root))
        if latent is None:
            raise ValueError("clean stage missing completed latent")
        if not completed_asset(directory, row["source_id"]+".png", clean_identity(run,row,root,latent)):
            missing.append((row, latent))
    if not missing:
        announce(stage="clean", reused=len(rows))
        return
    import numpy as np
    import torch
    configure_torch("cuda")
    from .adapter import load_adapter, pil_image, rgb8
    adapter = load_adapter(root/"cache"/"vae", Config(), "cuda")
    for row, latent in missing:
        name = row["source_id"]+".npy"
        if completed_asset(root/"cache"/"latent", name, source_identity(run,row,root)) is None:
            raise ValueError("clean stage missing completed latent")
        z = torch.from_numpy(np.load(root/"cache"/"latent"/name, allow_pickle=False))[None].to("cuda")
        with torch.no_grad():
            image = pil_image(rgb8(adapter.decode(z))[0])
        write_asset(directory, row["source_id"]+".png", clean_identity(run,row,root,latent), lambda p: image.save(p, format="PNG"))
        announce(stage="clean", source=row["source_id"], status="completed")


def manifest(root, run, data):
    from .data import Cache
    model = read_json(root/"model.json")
    units = []
    for row in data["sources"]:
        stem = row["source_id"]
        latent = completed_asset(root/"cache"/"latent", stem+".npy", source_identity(run,row,root))
        if latent is None:
            raise ValueError("incomplete latent: " + stem)
        clean_meta = completed_asset(root/"cache"/"clean", stem+".png", clean_identity(run,row,root,latent))
        if clean_meta is None:
            raise ValueError("incomplete source: " + stem)
        unit = {k:row[k] for k in ("source_id", "source_group", "template_family", "split", "reference_nonce")}
        unit.update({"unit_id": stem, "source_sha256": json_hash(row), "key": data["keys"][row["split"]][row["key_index"]],
                     "latent": "latent/"+stem+".npy", "latent_sha256": latent["sha256"],
                     "clean": "clean/"+stem+".png", "clean_sha256": clean_meta["sha256"],
                     "provenance": {"generation_seed": row["generation_seed"], "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                                    "sampler": "FlowMatchEulerDiscreteScheduler; full config in model inventory", "steps": 40, "guidance_scale": 4.5,
                                    "max_sequence_length": 256, "diffusers": "0.35.1", "model_revision": GENERATION["revision"]}})
        units.append(unit)
    value = {"schema": "self-reference-cache-v1", "config_hash": Config().digest, "execution_hash": ExecutionProfile().digest,
             "run_identity": run["identity"], "runtime": read_json(root/"runtime.json")["identity"],
             "latent_convention": "final_z0_scaled_before_decode", "keys": data["keys"], "units": units,
             "model": {"family": "SD3.5", "id": GENERATION["model_id"], "revision": GENERATION["revision"],
                       "vae_export": "vae", "vae_files": model["vae_files"], "denoiser_sha256": model["denoiser_sha256"],
                       "denoiser_files": model["denoiser_files"]}}
    path = root/"cache"/"manifest.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("existing manifest conflicts with completed source metadata")
    atomic_json(path, value)
    cache = Cache(path, Config())
    atomic_json(root/"cache_validation.json", cache.report())
    announce(stage="manifest", status="validated", **cache.report())


def checkpoint_plan(directory, cache, config, arm, execution):
    """Retain corrupt/advanced artifacts in quarantine; replay before any missing candidate."""
    from .engine import load_checkpoint
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    valid, bad = [], []
    for temporary in directory.glob("*.tmp"):
        quarantine([temporary], directory, "interrupted checkpoint temporary file")
    for path in sorted(directory.glob("*.pt")):
        try:
            state = load_checkpoint(path, cache, config, arm, "cpu")
            if state.get("execution_hash") != execution.digest:
                raise ValueError("execution identity conflict")
            valid.append((state["update"], path))
        except Exception as error:
            # Identity conflicts are not corruption: never silently restart a different experiment.
            if "identity" in str(error) or "execution" in str(error) or "version" in str(error):
                raise
            bad.append({"path": str(path), "error": redact(str(error))})
            quarantine([path, path.with_suffix(".json")], directory, redact(str(error)))
    for sidecar in directory.glob("*.json"):
        if sidecar.stem.startswith(("update_", "recovery_")) and not sidecar.with_suffix(".pt").exists():
            quarantine([sidecar], directory, "orphan checkpoint sidecar")
    required = set(range(config.checkpoint_every, config.updates+1, config.checkpoint_every))
    candidates = {u for u,p in valid if p.name == f"update_{u:06d}.pt"}
    if required <= candidates:
        result = {"status": "complete", "resume": None, "invalid": bad}
    else:
        hole = min(required-candidates)
        eligible = [(u,p) for u,p in valid if u < hole]
        last = max(eligible, default=(0,None), key=lambda x:x[0])
        for update, path in valid:
            if update > last[0]:
                quarantine([path,path.with_suffix(".json")], directory, "replay from before missing/corrupt fixed candidate")
        result = {"status": "resume" if last[1] else "start", "resume": str(last[1]) if last[1] else None, "update": last[0], "invalid": bad}
    atomic_json(directory/"recovery_plan.json", result)
    announce(stage="checkpoint_scan", arm=arm, **result)
    return result


def research_stage(root, task, arm=None):
    from .adapter import load_adapter, load_lpips
    from .data import Cache
    from .engine import train, select_checkpoints, validate_selection, evaluate
    config = Config()
    cache = Cache(root/"cache"/"manifest.json", config)
    if task == "train":
        plan = checkpoint_plan(root/"train"/arm, cache, config, arm, ExecutionProfile())
        if plan["status"] == "complete":
            return
        configure_torch("cuda")
        adapter = load_adapter(cache.model_dir, config, "cuda")
        train(cache, config, adapter, arm, root/"train"/arm, device="cuda", resume=plan["resume"],
              execution=ExecutionProfile(), progress=lambda values: announce(stage="train", **values))
        return
    selection = root/"selection.json"
    if task == "select":
        for a in ("B0", "B1", "C"):
            if checkpoint_plan(root/"train"/a, cache, config, a, ExecutionProfile())["status"] != "complete":
                raise ValueError("all three arms must finish before selection")
        if selection.exists():
            validate_selection(selection, cache, config)
            announce(stage="select", status="reused_locked_selection")
            return
        configure_torch("cuda")
        adapter = load_adapter(cache.model_dir, config, "cuda")
        select_checkpoints(cache, config, adapter, root/"train", selection, device="cuda")
        return
    validate_selection(selection, cache, config)
    output = root/(task+".json")
    weights = root/"quality"/"lpips-full.pt"
    metric_hash = file_sha(weights) if weights.exists() else None
    if task == "test" and not read_json(root/"dev.json")["A_gate"]["pass"]:
        raise ValueError("test cannot run after failed dev A")
    if output.exists():
        report = read_json(output)
        if report.get("manifest_hash") != cache.hash or report.get("config_hash") != config.digest or report.get("selection_hash") != file_sha(selection) or report.get("lpips",{}).get("sha256") != metric_hash or report.get("split") != task:
            raise ValueError("existing evaluation identity mismatch")
        announce(stage=task, status="reused_locked_report", research_status=report["research_status"])
        return
    configure_torch("cuda")
    adapter = load_adapter(cache.model_dir, config, "cuda")
    metric = load_lpips(weights, "cuda") if weights.exists() else None
    report = evaluate(cache, config, adapter, selection, task, output, device="cuda", metric=metric,
                      dev_report=root/"dev.json" if task == "test" else None,
                      metric_info={"status": "official_local_full_weights" if metric else "unavailable", "sha256": metric_hash})
    announce(stage=task, research_status=report["research_status"], A_gate=report["A_gate"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("hardware", "cache_status", "prepare", "prepare_quality", "generate", "clean", "manifest", "train", "select", "dev", "test"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--arm", choices=("B0", "B1", "C"))
    args = parser.parse_args(argv)
    root = Path(args.run)
    try:
        run, data = context(root)
        if args.task == "hardware":
            hardware(root, run)
        elif args.task == "cache_status":
            cache_status(root, run, data)
        elif args.task == "prepare":
            prepare_model(root, run)
            prepare_lpips(root)
        elif args.task == "prepare_quality":
            prepare_lpips(root)
        elif args.task in ("generate", "clean"):
            globals()[args.task](root, run, data["sources"][args.start:args.start+args.count])
        elif args.task == "manifest":
            manifest(root, run, data)
        else:
            research_stage(root, args.task, args.arm)
    except BaseException as error:
        safe = {"task": args.task, "status": "interrupted_or_error", "error": redact(f"{type(error).__name__}: {error}")}
        atomic_json(root/"worker_error.json", safe)
        announce(**safe)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
