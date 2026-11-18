"""Model-free parent process: persistent stages and redacted subprocess tee logs."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .colab_config import GENERATION, json_hash, roster
from .colab_storage import atomic_json, read_json, redact
from .config import Config
from .execution import ExecutionProfile


def initialize(repo, base, python, code_commit):
    repo, base = Path(repo).resolve(), Path(base).resolve()
    current = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if current != code_commit:
        raise ValueError("checkout differs from notebook's immutable code commit")
    paths = ["src/cegwm/self_reference", "configs/self_reference_v1_sources.json"]
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *paths]).returncode:
        raise ValueError("refusing modified execution code or source roster")
    _, sources_hash = roster(repo)
    identity = {"code_commit": code_commit, "config_hash": Config().digest,
                "generator_hash": json_hash(GENERATION), "sources_hash": sources_hash,
                "model_revision": GENERATION["revision"], "execution_hash": ExecutionProfile().digest}
    root = base/"runs"/json_hash(identity)[:24]
    root.mkdir(parents=True, exist_ok=True)
    path = root/"run.json"
    if path.exists() and read_json(path)["identity"] != identity:
        raise ValueError("run identity conflict")
    descriptor = {"identity": identity, "repo": str(repo), "base": str(base), "python": str(python),
                  "root": str(root), "generation": GENERATION}
    atomic_json(path, descriptor)
    return root


def run_logged(command, log, env=None):
    """No shell, no environment dump, no token command argument or unredacted tee."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as output:
        print("Starting:", redact(" ".join(map(str, command))), flush=True)
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env, start_new_session=True)
        try:
            for line in process.stdout:
                safe = redact(line)
                print(safe, end="", flush=True)
                output.write(safe)
            code = process.wait()
            if code:
                raise RuntimeError(f"worker exited with code {code}; see {log}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


class Controller:
    def __init__(self, root, invoke=None):
        self.root = Path(root)
        self.run = read_json(self.root/"run.json")
        self.ok, self.stopped = True, False
        self.invoke = invoke or self._invoke

    def _invoke(self, task, *args):
        env = os.environ.copy()
        env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": str(Path(self.run["repo"])/"src"), "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TOKENIZERS_PARALLELISM": "false", "CEGWM_PROGRESS": "1"})
        command = [self.run["python"], "-m", "cegwm.self_reference.colab_worker", task, "--run", str(self.root), *map(str,args)]
        run_logged(command, self.root/"logs"/(task+".log"), env)

    def _step(self, task, *args):
        atomic_json(self.root/"progress.json", {"status": "running", "task": task, "arguments": list(args), "time": time.time()})
        self.invoke(task, *args)
        atomic_json(self.root/"progress.json", {"status": "stage_completed", "task": task, "time": time.time()})

    def execute(self, stage):
        if not self.ok or self.stopped:
            print(f"SKIPPED {stage}: prior error/interruption or dev A stop", flush=True)
            return
        try:
            if stage == "cache":
                self._step("hardware")
                self._step("cache_status")
                cache_status = read_json(self.root/"cache_status.json")
                self._step("prepare" if cache_status["missing_latent"] or cache_status["missing_vae"] else "prepare_quality")
                if cache_status["missing_latent"]:
                    self._step("generate", "--count", 640)
                if cache_status["missing_clean"]:
                    self._step("clean", "--count", 640)
                self._step("manifest")
            elif stage == "train":
                for arm in ("B0", "B1", "C"):
                    self._step("train", "--arm", arm)
            elif stage == "evaluate":
                self._step("select")
                self._step("dev")
                dev = read_json(self.root/"dev.json")
                if not dev["A_gate"]["pass"]:
                    self.stopped = True
                    atomic_json(self.root/"progress.json", {"status": "stopped_by_dev_A", "reason": dev["A_gate"]["status"]})
                    print("STOP: dev A failed; shifted evaluation/test are not dispatched. Training augmentation was unchanged.", flush=True)
                    return
                self._step("test")
            else:
                raise ValueError("unknown notebook stage")
        except BaseException as error:
            self.ok = False
            reason = redact(f"{type(error).__name__}: {error}")
            atomic_json(self.root/"progress.json", {"status": "interrupted_or_error", "stage": stage, "reason": reason})
            print("STOP:", reason, "Rerun all to resume verified artifacts.", flush=True)

    def summary(self):
        progress = read_json(self.root/"progress.json") if (self.root/"progress.json").exists() else {"status": "not_started"}
        result = {"run": str(self.root), "identity": self.run["identity"], "progress": progress,
                  "status": "incomplete", "scientific_status": "not_evaluable"}
        for stage in ("dev", "test"):
            path = self.root/(stage+".json")
            if path.exists():
                report = read_json(path)
                result[stage] = {k: report[k] for k in ("A", "A_gate", "quality", "research_status")}
                # Do not repeat raw per-image quality arrays in notebook output.
                result[stage]["quality"] = {"status": report["quality"]["status"]}
                b = report.get("B", {})
                result[stage]["B"] = {k:b[k] for k in ("status", "source_denominator", "cluster_denominator", "gain_target_met", "complete") if k in b}
                if "arms" in b:
                    result[stage]["B"]["arms"] = {arm: {k:values[k] for k in ("all_four_source_rate", "all_four_source_count", "source_bootstrap95")}
                                                    for arm,values in b["arms"].items()}
                    result[stage]["B"]["paired"] = b["paired"]
        if progress["status"] == "stopped_by_dev_A":
            result["status"] = "stopped_by_dev_A"
        elif self.ok and "test" in result and progress.get("task") == "test" and progress["status"] == "stage_completed":
            result["status"] = "execution_completed"
            result["scientific_status"] = result["test"]["research_status"]
        atomic_json(self.root/"summary.json", result)
        return result
