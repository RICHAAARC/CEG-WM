"""Build an unexecuted Colab notebook with its small, exact source snapshot.

This is packaging, not a model run. It needs only the Python standard library.
No unpublished Git ref, code upload, runtime checkout edit or hash admission
gate is required by the resulting notebook. Rebuild after source edits.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/channel_precompensation_v1_colab.ipynb"
CONTINUATION_OUTPUT = ROOT / "notebooks/channel_precompensation_v1_continue_colab.ipynb"
SOURCE_FILES = (
    "configs/channel_precompensation_v1_colab.json",
    "requirements-reconstruction-channel-colab.txt",
    "experiments/run_channel_precompensation_v1.py",
    "experiments/run_reconstruction_channel_v1.py",
    "src/cegwm/__init__.py",
    "src/cegwm/channel_precompensation.py",
    "src/cegwm/reconstruction_channel.py",
    "src/cegwm/runtime/__init__.py",
    "src/cegwm/runtime/observation.py",
    "src/cegwm/runtime/reconstruction_channel_sd35.py",
    "src/cegwm/runtime/channel_precompensation_sd35.py",
    "src/cegwm/runtime/channel_precompensation_provider.py",
    "src/cegwm/runtime/channel_precompensation_resume.py",
    "src/cegwm/shared/__init__.py",
    "src/cegwm/shared/prg.py",
    "src/cegwm/shared/keys.py",
    "src/cegwm/shared/bands.py",
    "src/cegwm/shared/numerics.py",
)


def source_snapshot():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 8, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (ROOT / name).read_bytes())
    return buffer.getvalue()


HELPERS = '''import base64
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version

RUN_ROOT = Path("/content/drive/MyDrive/CEG-WM/channel-precompensation-v1/precompensation-20261108")
CONFIG_RELATIVE = Path("configs/channel_precompensation_v1_colab.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\\n", encoding="utf-8")
    temporary.replace(path)


def unpack_source(payload, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\\\" in info.filename:
                raise ValueError("Source archive contains an unsafe path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Source archive must not contain symlinks")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Source archive escapes its destination")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
    return destination


def terminal_counts(summary):
    stages = summary.get("stages", {})
    return bool(stages and not any(value == "started" for value in stages.values())
                and stages.get("cleanup") in {"complete", "failed", "no_provider"})


def ensure_runtime(checkout, run_root):
    import torch
    parsed = re.match(r"(\\d+)\\.(\\d+)", torch.__version__)
    metadata = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    if metadata["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        metadata.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory)
    write_json(run_root / "environment.json", metadata)
    if parsed is None or tuple(map(int, parsed.groups())) < (2, 6) or not metadata["cuda_available"]:
        raise RuntimeError("Use a Colab GPU runtime with CUDA PyTorch >=2.6. PyTorch/CUDA will not be reinstalled.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(checkout / "requirements-reconstruction-channel-colab.txt")], check=True)
    for package in ("diffusers", "transformers", "accelerate", "sentencepiece", "protobuf", "numpy", "Pillow"):
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = None
    write_json(run_root / "environment.json", metadata)
    print(metadata)
    from google.colab import userdata
    token = os.environ.get("HF_TOKEN") or userdata.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN to Colab Secrets before a new run.")
    environment = dict(os.environ)
    environment["HF_TOKEN"] = token
    environment["PYTHONPATH"] = str(checkout / "src") + os.pathsep + str(checkout)
    return environment


def run_or_show(config, checkout, run_root, payload):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    configuration = run_root / "config.json"
    if configuration.exists() and read_json(configuration) != config:
        raise RuntimeError("Existing run configuration differs; evidence retained and no computation started.")
    if not configuration.exists():
        write_json(configuration, config)
    output = run_root / "batch"
    if output.exists():
        print({"action": "read_only_existing", "output": str(output),
               "note": "Completed, failed and interrupted runs are all retained without automatic retry."})
        return read_json(output / "summary.json")
    # With no physical batch yet, this is the snapshot for the forthcoming
    # attempt, including when environment setup subsequently fails.
    (run_root / "source_snapshot.zip").write_bytes(payload)
    environment = ensure_runtime(checkout, run_root)
    command = [sys.executable, "-m", "experiments.run_channel_precompensation_v1",
               "--config", str(checkout / CONFIG_RELATIVE), "--execute", "--output", str(output)]
    try:
        completed = subprocess.run(command, cwd=checkout, env=environment, check=False)
    except BaseException as error:
        write_json(run_root / "launch_error.json", {"error": type(error).__name__ + ": " + str(error)})
        raise
    if not (output / "summary.json").exists():
        raise RuntimeError("Runner produced no readable summary; inspect saved errors and do not rerun automatically.")
    summary = read_json(output / "summary.json")
    print({"runner_exit": completed.returncode, "method_summaries": summary.get("method_summaries", {}),
           "note": "An unreachable write remains in the denominator; a nonzero exit is not permission to retry."})
    return summary


def package_results(config, run_root):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "batch/summary.json"
    try:
        summary = read_json(summary_path) if summary_path.exists() else {}
    except (OSError, ValueError) as error:
        summary = {"read_error": str(error)}
    rows = {row["source_id"]: row for row in summary.get("sources", [])}
    terminal = terminal_counts(summary)
    result = {"scope": "four_source_one_shot_falsification_only", "config": config,
              "counts_terminal": terminal,
              "actual_calls": summary.get("actual_calls") if terminal else None,
              "last_recorded_calls": summary.get("actual_calls", {}),
              "method_summaries": summary.get("method_summaries", {}),
              "sources": [rows.get(row["source_id"], {"source_id": row["source_id"], "status": "not_observed"})
                          for row in config["sources"]],
              "claim": "No scientific PASS, watermark-presence, attribution, FPR or general robustness conclusion."}
    write_json(run_root / "measurement_summary.json", result)
    destination = run_root.parent / (run_root.name + "-results.zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_root.resolve()):
                archive.write(path, Path(run_root.name) / path.relative_to(run_root))
    return result, destination
'''


def cell(identifier, kind, source):
    result = {"cell_type": kind, "id": identifier, "metadata": {},
              "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(payload=None):
    payload = source_snapshot() if payload is None else payload
    lines = textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100)
    embedded = "#@title 本次源码快照（无需上传代码包）\nSOURCE_BUNDLE_B64 = (\n" + "".join(
        "    " + repr(line) + "\n" for line in lines) + ")\n"
    cells = [
        cell("intro", "markdown", "# 单次双通道预补偿 · 最小否定验证\n\n"
             "本 notebook 内含本次源码及固定配置，无需公开仓库发布、手工填写版本或上传代码包。"
             "在 Colab 打开本文件，选择 GPU、配置有模型访问权限的 HF_TOKEN Secret，再 Run all。\n\n"
             "固定 4 个新来源、两个比特、联合区间法与同信息单步对照。质量只报告，不设质量失败门槛。"
             "未标记图、不可达、读错和运行失败均保留；已有结果只读，不自动重试。\n\n"
             "最多 20 张正式生成图、40 次正式重建、60 次公开读出；另有 4 次求导重建前向、"
             "4 次源末段前向、8 次潜变量 VJP、8 次 RGB VJP 及相关读者前向。"
             "完整重建前向合计最多 44 次，实际调用分别记录。CPU/fake 检查不代表真实 GPU 或方法成功。\n"),
        cell("drive_mount", "code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
        cell("helpers", "code", HELPERS),
        cell("embedded_source", "code", embedded),
        cell("prepare_source", "code", '''SOURCE_PAYLOAD = base64.b64decode(SOURCE_BUNDLE_B64)
CHECKOUT = unpack_source(SOURCE_PAYLOAD, Path(tempfile.mkdtemp(prefix="cegwm-precomp-")) / "source")
CONFIG = read_json(CHECKOUT / CONFIG_RELATIVE)
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
print({"source": "embedded_snapshot", "output": str(RUN_ROOT), "quality": "report_only"})
'''),
        cell("run_info", "markdown", "## 执行固定名册\n\n"
             "模型按阶段装载，SD3 与 SDXL 大组件分时使用。求导阶段移除推理 offload hooks，"
             "激活保存至 CPU；实际显存和主机内存占用尚未通过真实 GPU 验证。"
             "若中止，保留已有文件并运行末尾打包格，不修改名册或更换目录重试。\n"),
        cell("execute", "code", '''try:
    SUMMARY = run_or_show(CONFIG, CHECKOUT, RUN_ROOT, SOURCE_PAYLOAD)
except Exception as error:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_ROOT / "notebook_error.json", {"error": type(error).__name__ + ": " + str(error)})
    print({"execution_stopped": type(error).__name__ + ": " + str(error), "evidence_retained": str(RUN_ROOT)})
finally:
    AGGREGATE, RESULTS_ZIP = package_results(CONFIG, RUN_ROOT)
'''),
        cell("return_info", "markdown", "## 汇总与回传\n\n本格只读取和打包已有文件，不调用模型。"
             "可在上一格报错后单独运行。回传下载的 ZIP；其中保留源码快照、原始图像／特征、"
             "写端敏感度和实际扰动、逐项失败及调用计数。中断时的计数只标为最后记录，不能当完成总数。\n"),
        cell("package", "code", '''AGGREGATE, RESULTS_ZIP = package_results(CONFIG, RUN_ROOT)
print(json.dumps({"counts_terminal": AGGREGATE["counts_terminal"],
                  "actual_calls": AGGREGATE["actual_calls"],
                  "method_summaries": AGGREGATE["method_summaries"],
                  "sources": [{"source_id": row["source_id"], "status": row.get("status")}
                              for row in AGGREGATE["sources"]],
                  "results_zip": str(RESULTS_ZIP)}, indent=2, ensure_ascii=False))
from google.colab import files
files.download(str(RESULTS_ZIP))
'''),
    ]
    cells[3]["metadata"]["cellView"] = "form"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []}}, "cells": cells}


CONTINUATION_HELPERS = '''
RESUME_NAME = "continuation_layout_v2"
RESUME_BATCH = RUN_ROOT / RESUME_NAME
CONTINUATION_NAME = "continuation_suffix_device_v1"
CONTINUATION_LAUNCH_METADATA = None


def new_continuation_metadata(run_root):
    root = Path(run_root) / (CONTINUATION_NAME + "_metadata")
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = root / ("launch_" + str(index).zfill(4))
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def run_or_show_continuation(checkout, run_root, payload):
    global CONTINUATION_LAUNCH_METADATA
    CONTINUATION_LAUNCH_METADATA = None
    run_root = Path(run_root)
    original = run_root / RESUME_NAME
    if not (original / "config.json").is_file() or not (original / "summary.json").is_file():
        raise RuntimeError("The saved continuation_layout_v2 checkpoint is required; no zero images will be regenerated.")
    output = run_root / CONTINUATION_NAME
    if output.exists():
        print({"action": "read_only_existing_continuation", "output": str(output)})
        return read_json(output / "summary.json")
    metadata_dir = new_continuation_metadata(run_root)
    CONTINUATION_LAUNCH_METADATA = metadata_dir
    (metadata_dir / "source_snapshot.zip").write_bytes(payload)
    # Keep the original environment/config/source snapshot and failed batch.
    environment = ensure_runtime(checkout, metadata_dir)
    command = [sys.executable, "-u", "-m", "experiments.run_channel_precompensation_v1",
               "--resume-from", str(original), "--execute", "--output", str(output)]
    log_path = metadata_dir / "runner.log"
    try:
        with log_path.open("w", encoding="utf-8") as runner_log:
            with subprocess.Popen(command, cwd=checkout, env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    runner_log.write(line)
                    runner_log.flush()
                returncode = process.wait()
        write_json(metadata_dir / "runner_process.json",
                   {"returncode": returncode, "log_path": str(log_path)})
    except BaseException as error:
        write_json(metadata_dir / "launch_error.json",
                   {"error": type(error).__name__ + ": " + str(error), "log_path": str(log_path)})
        raise
    try:
        summary = read_json(output / "summary.json")
    except (OSError, ValueError) as error:
        raise RuntimeError("Continuation produced no readable summary; runner return code "
                           + str(returncode) + "; log: " + str(log_path)
                           + "; original evidence is retained.") from error
    print({"runner_exit": returncode, "runner_log": str(log_path),
           "method_summaries": summary.get("method_summaries", {}),
           "note": "Saved zero observations and compatible RGB gradients are reused; failed or unreachable writes remain in the denominator."})
    return summary


def package_continuation(config, run_root):
    run_root = Path(run_root)
    summary_path = run_root / CONTINUATION_NAME / "summary.json"
    try:
        summary = read_json(summary_path) if summary_path.is_file() else {}
    except (OSError, ValueError) as error:
        summary = {"read_error": str(error)}
    original = read_json(run_root / RESUME_NAME / "summary.json")
    terminal = terminal_counts(summary)
    rows = {row["source_id"]: row for row in summary.get("sources", [])}
    report = {"scope": "suffix_device_fix_continuation_of_saved_rgb_gradients", "config": config,
              "resume_from": str(run_root / RESUME_NAME),
              "counts_terminal": terminal,
              "prior_actual_calls": original.get("cumulative_actual_calls", original.get("actual_calls", {})),
              "actual_calls": summary.get("actual_calls") if terminal else None,
              "last_recorded_calls": summary.get("actual_calls", {}),
              "cumulative_actual_calls": summary.get("cumulative_actual_calls") if terminal else None,
              "reused_counts": summary.get("reused_counts", {}),
              "method_summaries": summary.get("method_summaries", {}),
              "sources": [rows.get(row["source_id"], {"source_id": row["source_id"], "status": "not_observed"})
                          for row in config["sources"]],
              "claim": "Engineering continuation; original failure retained; no scientific PASS."}
    write_json(run_root / (CONTINUATION_NAME + "_summary.json"), report)
    destination = run_root.parent / (run_root.name + "-continuation-suffix-device-v1-results.zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_root.resolve()):
                archive.write(path, Path(run_root.name) / path.relative_to(run_root))
    return report, destination
'''


def build_continuation_notebook(payload=None):
    notebook = build_notebook(payload)
    replacements = {
        "intro": ("# 预补偿验证：复用 RGB 梯度继续源末段\n\n"
            "本入口修正 SD3 源末段恢复时调度器 sigmas 的设备位置，使其与官方生成路径一致。"
            "保留逐像素精确前向检查、读者、码格、预算、四个来源和两个重建种子，质量仍只报告。"
            "真实 L4 上的修复效果尚未验证。\n\n"
            "从 Drive 中 precompensation-20261108/continuation_layout_v2 恢复 4 个未标记捕获点、"
            "12 份零臂 RGB/特征及已保存的 8 条 RGB 梯度。当前完整缓存直接复用，不重新生成零臂、"
            "重算 RGB VJP 或运行 SDXL 代理重建。原 batch、旧续跑和全部启动日志保持原样。\n\n"
            "本次剩余 4 次源末段前向及 8 次潜变量 VJP；全部可写时，另新增最多 16 次正式生成、"
            "32 次正式重建、48 次公开读出。通用 runner 的预算仍包括缓存缺失时最多 8 条 RGB VJP"
            "和 4 次代理重建的保守上限；本次 8 条兼容缓存齐全时，这两项预期新增调用均为 0。"
            "旧调用从输入 checkpoint 的累计计数接续，本次新增与累计分别报告。\n\n"
            "新结果仅写入 continuation_suffix_device_v1，启动输出与报错完整存入 ZIP；"
            "metadata 下的 launch 编号只区分保存的启动诊断，不改变执行目录或重试规则。\n\n"
            "在 Colab 选择 GPU（上次使用 L4），设置 HF_TOKEN 后 Run all。首格仅挂载 Drive。"
            "已完成或已中断的本次续跑仍只读，不自动再次尝试。\n"),
        "helpers": HELPERS + CONTINUATION_HELPERS,
        "prepare_source": '''SOURCE_PAYLOAD = base64.b64decode(SOURCE_BUNDLE_B64)
CHECKOUT = unpack_source(SOURCE_PAYLOAD, Path(tempfile.mkdtemp(prefix="cegwm-precomp-continue-")) / "source")
CONFIG = read_json(RESUME_BATCH / "config.json")
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
print({"resume_from": str(RESUME_BATCH), "new_output": str(RUN_ROOT / CONTINUATION_NAME),
       "quality": "report_only", "regenerate_zero_images": False,
       "expected_reused_rgb_gradients": 8, "expected_new_rgb_vjps": 0,
       "expected_new_proxy_reconstructions": 0})
''',
        "run_info": ("## 执行剩余步骤\n\n原零臂图像、RGB 梯度、配置和失败文件保持原样。"
            "新梯度及失败前诊断逐步保存。若仍有前向不一致，会保留证据并停止相应来源，"
            "不会放宽门槛、调参、换来源或隐式重试。\n"),
        "execute": '''try:
    SUMMARY = run_or_show_continuation(CHECKOUT, RUN_ROOT, SOURCE_PAYLOAD)
except Exception as error:
    metadata_dir = CONTINUATION_LAUNCH_METADATA or new_continuation_metadata(RUN_ROOT)
    write_json(metadata_dir / "notebook_error.json", {"error": type(error).__name__ + ": " + str(error)})
    print({"execution_stopped": type(error).__name__ + ": " + str(error), "original_retained": str(RESUME_BATCH)})
finally:
    AGGREGATE, RESULTS_ZIP = package_continuation(CONFIG, RUN_ROOT)
''',
        "return_info": ("## 汇总与回传\n\n本格仅打包现有文件，包含原失败批次和本次续跑。"
            "无需重新运行上面的模型格；回传下载的 ZIP 即可。\n"),
        "package": '''AGGREGATE, RESULTS_ZIP = package_continuation(CONFIG, RUN_ROOT)
print(json.dumps({"counts_terminal": AGGREGATE["counts_terminal"],
                  "prior_actual_calls": AGGREGATE["prior_actual_calls"],
                  "actual_calls": AGGREGATE["actual_calls"],
                  "cumulative_actual_calls": AGGREGATE["cumulative_actual_calls"],
                  "reused_counts": AGGREGATE["reused_counts"],
                  "method_summaries": AGGREGATE["method_summaries"],
                  "sources": [{"source_id": row["source_id"], "status": row.get("status")}
                              for row in AGGREGATE["sources"]],
                  "results_zip": str(RESULTS_ZIP)}, indent=2, ensure_ascii=False))
from google.colab import files
files.download(str(RESULTS_ZIP))
''',
    }
    for entry in notebook["cells"]:
        if entry["id"] in replacements:
            entry["source"] = replacements[entry["id"]].splitlines(keepends=True)
    notebook["metadata"]["colab"]["name"] = CONTINUATION_OUTPUT.name
    return notebook


if __name__ == "__main__":
    payload = source_snapshot()
    for path, notebook in ((OUTPUT, build_notebook(payload)),
                           (CONTINUATION_OUTPUT, build_continuation_notebook(payload))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)
