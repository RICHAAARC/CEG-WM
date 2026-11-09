"""Build the independent, unexecuted source05 method-validation notebook.

The source snapshot is embedded; publishing or running this builder never loads
models. Existing production and diagnostic notebooks are not modified.
"""

from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile

from scripts.build_channel_precompensation_notebook import ROOT, SOURCE_FILES as BASE_SOURCE_FILES, cell
from scripts.build_channel_suffix_diagnostic_notebook import _common_helpers


OUTPUT = ROOT / "notebooks/channel_precompensation_single_source_v1_colab.ipynb"
SOURCE_FILES = (*BASE_SOURCE_FILES,
                "experiments/run_channel_precompensation_single_source_v1.py",
                "src/cegwm/runtime/channel_precompensation_single_source.py")


def source_snapshot():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (ROOT / name).read_bytes())
    return buffer.getvalue()


METHOD_HELPERS = '''
import traceback

RESUME_NAME = "continuation_suffix_device_v1"
METHOD_NAME = "single_source_method_v1"
SOURCE_ID = "source05_smooth_stilllife"
METHODS = ("joint", "gradient")
BITS = (0, 1)
METHOD_LAUNCH_METADATA = None


def new_method_metadata(run_root):
    root = Path(run_root) / (METHOD_NAME + "_metadata")
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = root / ("launch_" + str(index).zfill(4))
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def optional_json(path):
    path = Path(path)
    if not path.is_file():
        return {}, "missing: " + str(path)
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, None
    except (OSError, ValueError) as error:
        return {}, type(error).__name__ + ": " + str(error)


def run_or_show_single_source(checkout, run_root, payload):
    global METHOD_LAUNCH_METADATA
    METHOD_LAUNCH_METADATA = None
    run_root = Path(run_root)
    output = run_root / METHOD_NAME
    if output.exists():
        summary, error = optional_json(output / "summary.json")
        print({"action": "read_only_existing_single_source", "output": str(output),
               "summary_error": error, "note": "Existing completed, failed or interrupted output is retained without retry."})
        return summary
    original = run_root / RESUME_NAME
    if not (original / "config.json").is_file() or not (original / "summary.json").is_file():
        raise RuntimeError("The saved continuation_suffix_device_v1 checkpoint is required before method validation.")
    metadata = new_method_metadata(run_root)
    METHOD_LAUNCH_METADATA = metadata
    (metadata / "source_snapshot.zip").write_bytes(payload)
    command = [sys.executable, "-u", "-m", "experiments.run_channel_precompensation_single_source_v1",
               "--resume-from", str(original), "--execute", "--output", str(output)]
    write_json(metadata / "launch.json", {"command": command, "resume_from": str(original),
               "output": str(output), "source_id": SOURCE_ID, "methods": list(METHODS), "bits": list(BITS)})
    environment = ensure_runtime(checkout, metadata)
    log_path = metadata / "runner.log"
    process = None
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
        write_json(metadata / "runner_process.json", {"returncode": returncode,
                   "log_path": str(log_path), "streams": "stdout_and_stderr_combined"})
    except BaseException as error:
        returncode = None if process is None else process.returncode
        write_json(metadata / "runner_process.json", {"returncode": returncode,
                   "log_path": str(log_path), "streams": "stdout_and_stderr_combined"})
        write_json(metadata / "launch_error.json", {"error": type(error).__name__ + ": " + str(error),
                   "traceback": traceback.format_exc(), "log_path": str(log_path)})
        raise
    summary, error = optional_json(output / "summary.json")
    if error:
        raise RuntimeError("Single-source runner produced no readable summary; runner return code " + str(returncode)
                           + "; counts unknown; log: " + str(log_path))
    print({"runner_exit": returncode, "runner_log": str(log_path),
           "method_summaries": summary.get("method_summaries", {}),
           "note": "Only source05 is tested. Collection status does not establish general robustness, attribution or FPR."})
    return summary


def package_single_source(run_root):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    original = run_root / RESUME_NAME
    summary, summary_error = optional_json(run_root / METHOD_NAME / "summary.json")
    prior, prior_error = optional_json(original / "summary.json")
    terminal = terminal_counts(summary) and isinstance(summary.get("actual_calls"), dict)
    report = {"scope": "source05_single_source_method_validation",
              "resume_from": str(original), "source_id": SOURCE_ID,
              "methods": list(METHODS), "bits": list(BITS),
              "counts_terminal": terminal, "new_counts_status": "terminal" if terminal else "unknown",
              "prior_calls_scope": summary.get("prior_calls_scope", "saved four-source checkpoint cumulative ledger"),
              "new_calls_scope": "source05 only",
              "prior_actual_calls": summary.get("prior_actual_calls", prior.get("cumulative_actual_calls", prior.get("actual_calls"))),
              "actual_calls": summary.get("actual_calls") if terminal else None,
              "last_recorded_calls": summary.get("actual_calls", {}),
              "cumulative_actual_calls": summary.get("cumulative_actual_calls") if terminal else None,
              "summary_error": summary_error, "prior_summary_error": prior_error,
              "method_summaries": summary.get("method_summaries", {}),
              "method_summary": summary,
              "original_evidence": "Original four-source checkpoint remains unchanged at resume_from; other three sources are untested in this run.",
              "claim": "One source, two bits, fixed reader and two reconstruction seeds; no general robustness, watermark-presence, attribution or FPR conclusion."}
    report_path = run_root / (METHOD_NAME + "_summary.json")
    write_json(report_path, report)
    destination = run_root.parent / (run_root.name + "-single-source-method-v1-results.zip")
    included_roots = [run_root / METHOD_NAME, run_root / (METHOD_NAME + "_metadata")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, Path(run_root.name) / report_path.relative_to(run_root))
        for directory in included_roots:
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()):
                    if path.suffix.lower() == ".zip" and path.name != "source_snapshot.zip":
                        continue
                    archive.write(path, Path(run_root.name) / path.relative_to(run_root))
    return report, destination
'''

HELPERS = _common_helpers() + METHOD_HELPERS


def build_notebook(payload=None):
    payload = source_snapshot() if payload is None else payload
    lines = textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100)
    embedded = "#@title 单源方法验证源码（无需上传或填写 Git 版本）\nSOURCE_BUNDLE_B64 = (\n" + "".join(
        "    " + repr(line) + "\n" for line in lines) + ")\n"
    cells = [
        cell("intro", "markdown", "# 单源预补偿 · 方法验证\n\n"
             "固定 source05_smooth_stilllife，比较 joint 与 gradient 两种写法，各写入 bit 0 和 bit 1。"
             "复用已有捕获点、读者和配置，在当前 suffix 路径建立新的 zero 基点，"
             "重新计算 clean／design 两条 RGB 梯度；旧梯度保留，不跨基点使用。"
             "历史 zero 图像差异仅作诊断，另外三个来源本轮不测。\n\n"
             "最多 6 次 suffix 前向／末步／decode（zero 1、潜变量 VJP 前向 1、写入 4），"
             "2 次潜变量 VJP、2 次 RGB VJP；10 次正式 SDXL 重建加 1 次代理重建，"
             "合计最多 11 次重建、15 次公开读出，完整生成次数为 0。"
             "四个写入臂直接比较预测变化与真实读数变化，不追加探针、拟合、调参或重试。"
             "质量只报告。单源结果不证明一般重建鲁棒性、归属或 FPR。\n\n"
             "读取 Drive 中 precompensation-20261108/continuation_suffix_device_v1，"
             "新结果固定写入 single_source_method_v1。已有输出只读，原证据不覆盖。"
             "源码内嵌，无需上传 ZIP 或手填 GitRef。\n\n"
             "在 Colab 选择 GPU，设置有模型访问权限的 HF_TOKEN Secret 后 Run all。"
             "首个代码格仅挂载 Drive。启动日志保留 stdout、stderr 和返回码；"
             "缺少可读 summary 时，新增计数显示 unknown，不推断为零。"
             "本入口只完成 CPU／静态检查，真实模型执行由你运行。\n"),
        cell("drive_mount", "code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
        cell("helpers", "code", HELPERS),
        cell("embedded_source", "code", embedded),
        cell("prepare_source", "code", '''SOURCE_PAYLOAD = base64.b64decode(SOURCE_BUNDLE_B64)
CHECKOUT = unpack_source(SOURCE_PAYLOAD, Path(tempfile.mkdtemp(prefix="cegwm-single-source-")) / "source")
print({"resume_from": str(RUN_ROOT / RESUME_NAME), "output": str(RUN_ROOT / METHOD_NAME),
       "source_id": SOURCE_ID, "methods": list(METHODS), "bits": list(BITS),
       "old_rgb_gradients_reused": False, "new_zero_reference": "current_suffix",
       "maximum_new_calls": {"suffix_forward": 6, "denoising_step": 6, "vae_decode": 6,
                             "full_generation": 0, "ordinary_sdxl": 10, "proxy_sdxl": 1,
                             "latent_vjp": 2, "rgb_vjp": 2, "public_read": 15}})
'''),
        cell("execute_info", "markdown", "## 执行固定单源\n\n"
             "runner 在加载模型前建立固定输出目录并保存初始 summary。"
             "已有结果只读；环境安装或启动失败日志追加保存。失败和不可达写入保留，"
             "不换来源、不新增 probe、不自动重试。\n"),
        cell("execute", "code", '''try:
    SUMMARY = run_or_show_single_source(CHECKOUT, RUN_ROOT, SOURCE_PAYLOAD)
except Exception as error:
    metadata = METHOD_LAUNCH_METADATA or new_method_metadata(RUN_ROOT)
    write_json(metadata / "notebook_error.json", {"error": type(error).__name__ + ": " + str(error),
               "traceback": traceback.format_exc()})
    print({"method_run_stopped": type(error).__name__ + ": " + str(error),
           "original_retained": str(RUN_ROOT / RESUME_NAME), "metadata": str(metadata)})
finally:
    REPORT, RESULTS_ZIP = package_single_source(RUN_ROOT)
'''),
        cell("package_info", "markdown", "## 汇总与回传\n\n"
             "本格只读取并打包已有结果，可在启动失败后单独运行。"
             "ZIP 包含单源输出、原始证据引用、源码、环境和启动日志；"
             "原批次继续保存在 Drive，不重复打包整个旧目录或外部 ZIP。"
             "原四源累计账目、本轮单源新增调用和合并累计分开报告。\n"),
        cell("package", "code", '''REPORT, RESULTS_ZIP = package_single_source(RUN_ROOT)
print(json.dumps({"counts_terminal": REPORT["counts_terminal"], "new_counts_status": REPORT["new_counts_status"],
                  "prior_actual_calls": REPORT["prior_actual_calls"], "actual_calls": REPORT["actual_calls"],
                  "cumulative_actual_calls": REPORT["cumulative_actual_calls"],
                  "method_summaries": REPORT["method_summaries"], "results_zip": str(RESULTS_ZIP)},
                 indent=2, ensure_ascii=False))
from google.colab import files
files.download(str(RESULTS_ZIP))
'''),
    ]
    cells[3]["metadata"]["cellView"] = "form"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []}}, "cells": cells}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
