"""Package the independent, unexecuted three-forward suffix diagnostic entry.

Uses only the standard library. It does not edit either production notebook or
execute a model. Rebuild after the diagnostic runner and runtime are finalized.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import textwrap
import zipfile

from scripts.build_channel_precompensation_notebook import (
    ROOT, SOURCE_FILES as BASE_SOURCE_FILES, HELPERS as BASE_HELPERS, cell,
)


OUTPUT = ROOT / "notebooks/channel_suffix_diagnostic_v1_colab.ipynb"
SOURCE_FILES = (*BASE_SOURCE_FILES, "experiments/run_channel_suffix_diagnostic_v1.py")


def source_snapshot():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 8, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (ROOT / name).read_bytes())
    return buffer.getvalue()


def _common_helpers():
    """Embed the existing setup/extraction helpers, excluding production launches."""
    functions = {"read_json", "write_json", "unpack_source", "terminal_counts", "ensure_runtime"}
    nodes = ast.parse(BASE_HELPERS).body
    selected = [node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))
                or isinstance(node, ast.FunctionDef) and node.name in functions]
    return "\n\n".join(ast.get_source_segment(BASE_HELPERS, node) for node in selected) + "\n"


DIAGNOSTIC_HELPERS = '''
import traceback

RESUME_NAME = "continuation_suffix_device_v1"
DIAGNOSTIC_NAME = "suffix_diagnostic_v1"
SOURCE_ID = "source05_smooth_stilllife"
ARM_ORDER = ("NG1", "NG2", "G")
DIAGNOSTIC_LAUNCH_METADATA = None


def new_diagnostic_metadata(run_root):
    root = Path(run_root) / (DIAGNOSTIC_NAME + "_metadata")
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


def run_or_show_diagnostic(checkout, run_root, payload):
    global DIAGNOSTIC_LAUNCH_METADATA
    DIAGNOSTIC_LAUNCH_METADATA = None
    run_root = Path(run_root)
    output = run_root / DIAGNOSTIC_NAME
    if output.exists():
        summary, error = optional_json(output / "summary.json")
        print({"action": "read_only_existing_diagnostic", "output": str(output),
               "summary_error": error, "note": "Existing completed, failed or interrupted outputs are never rerun."})
        return summary
    original = run_root / RESUME_NAME
    if not (original / "config.json").is_file() or not (original / "summary.json").is_file():
        raise RuntimeError("The saved continuation_suffix_device_v1 checkpoint is required; no zero generation is permitted.")
    metadata = new_diagnostic_metadata(run_root)
    DIAGNOSTIC_LAUNCH_METADATA = metadata
    (metadata / "source_snapshot.zip").write_bytes(payload)
    command = [sys.executable, "-u", "-m", "experiments.run_channel_suffix_diagnostic_v1",
               "--resume-from", str(original), "--execute", "--output", str(output)]
    write_json(metadata / "launch.json", {"command": command, "resume_from": str(original),
               "output": str(output), "source_id": SOURCE_ID, "arm_order": list(ARM_ORDER)})
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
        raise RuntimeError("Diagnostic produced no readable summary; runner return code " + str(returncode)
                           + "; counts unknown; log: " + str(log_path))
    print({"runner_exit": returncode, "runner_log": str(log_path),
           "classification": summary.get("classification"),
           "note": "Three-forward implementation diagnosis only; G has no backward and gives no scientific verdict."})
    return summary


def package_diagnostic(run_root):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    original = run_root / RESUME_NAME
    summary, summary_error = optional_json(run_root / DIAGNOSTIC_NAME / "summary.json")
    prior, prior_error = optional_json(original / "summary.json")
    terminal = terminal_counts(summary)
    report = {"scope": "fixed_source05_ng1_ng2_g_suffix_forward_diagnostic",
              "resume_from": str(original), "source_id": SOURCE_ID, "arm_order": list(ARM_ORDER),
              "counts_terminal": terminal, "new_counts_status": "terminal" if terminal else "unknown",
              "prior_actual_calls": prior.get("cumulative_actual_calls", prior.get("actual_calls")),
              "actual_calls": summary.get("actual_calls") if terminal else None,
              "last_recorded_calls": summary.get("actual_calls", {}),
              "cumulative_actual_calls": summary.get("cumulative_actual_calls") if terminal else None,
              "summary_error": summary_error, "prior_summary_error": prior_error,
              "classification": summary.get("classification"), "comparisons": summary.get("comparisons", {}),
              "diagnostic_summary": summary,
              "original_evidence": "Original runs remain unchanged at resume_from; required source05 reference inputs are copied by the runner.",
              "claim": "At most 3 suffix/step/decode forwards; no full generation, SDXL, VJP or public read. G is forward-only; no scientific validation."}
    report_path = run_root / (DIAGNOSTIC_NAME + "_summary.json")
    write_json(report_path, report)
    destination = run_root.parent / (run_root.name + "-suffix-diagnostic-v1-results.zip")
    included_roots = [run_root / DIAGNOSTIC_NAME, run_root / (DIAGNOSTIC_NAME + "_metadata")]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, Path(run_root.name) / report_path.relative_to(run_root))
        for directory in included_roots:
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()):
                    # Keep the small embedded source snapshot; never recursively
                    # wrap a previously downloaded results ZIP.
                    if path.suffix.lower() == ".zip" and path.name != "source_snapshot.zip":
                        continue
                    archive.write(path, Path(run_root.name) / path.relative_to(run_root))
    return report, destination
'''

HELPERS = _common_helpers() + DIAGNOSTIC_HELPERS


def build_notebook(payload=None):
    payload = source_snapshot() if payload is None else payload
    lines = textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100)
    embedded = "#@title 独立诊断源码（无需上传或填写 Git 版本）\nSOURCE_BUNDLE_B64 = (\n" + "".join(
        "    " + repr(line) + "\n" for line in lines) + ")\n"
    cells = [
        cell("intro", "markdown", "# 源末段差异诊断 · 独立三次前向\n\n"
             "固定 source05_smooth_stilllife，按 NG1、NG2、G 顺序运行。"
             "最多 3 次 suffix 前向、3 个末步、3 次 VAE decode；"
             "完整生成、SDXL、VJP、公开读出均为 0。G 仅开启梯度记录做前向，没有反向，"
             "用于区分前向数值差异，不是科学验证或方法成败判定。\n\n"
             "读取 Drive 中 precompensation-20261108/continuation_suffix_device_v1，"
             "新输出固定为 suffix_diagnostic_v1。已有输出只读，不自动重试或改目录；"
             "原证据不覆盖。源码内嵌，无需手填 GitRef 或上传 ZIP。\n\n"
             "在 Colab 选择 GPU、配置有模型访问权限的 HF_TOKEN Secret，再 Run all。"
             "首个代码格仅挂载 Drive。启动日志完整保留标准输出、错误输出及返回码；"
             "没有可读 summary 时，新增计数标为 unknown，不推断为零。\n"),
        cell("drive_mount", "code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
        cell("helpers", "code", HELPERS),
        cell("embedded_source", "code", embedded),
        cell("prepare_source", "code", '''SOURCE_PAYLOAD = base64.b64decode(SOURCE_BUNDLE_B64)
CHECKOUT = unpack_source(SOURCE_PAYLOAD, Path(tempfile.mkdtemp(prefix="cegwm-suffix-diagnostic-")) / "source")
print({"resume_from": str(RUN_ROOT / RESUME_NAME), "output": str(RUN_ROOT / DIAGNOSTIC_NAME),
       "source_id": SOURCE_ID, "arm_order": list(ARM_ORDER),
       "maximum_new_calls": {"suffix_forward": 3, "denoising_step": 3, "vae_decode": 3,
                             "full_generation": 0, "sdxl": 0, "vjp": 0, "public_read": 0}})
'''),
        cell("execute_info", "markdown", "## 执行诊断\n\n"
             "runner 在加载模型前创建固定输出目录并保存初始 summary。"
             "遇到已有输出只读取结果；环境安装或启动失败的日志追加保留。"
             "NG1／NG2 检查同一路径重复前向，G 与其比较梯度模式；不调用 backward。\n"),
        cell("execute", "code", '''try:
    SUMMARY = run_or_show_diagnostic(CHECKOUT, RUN_ROOT, SOURCE_PAYLOAD)
except Exception as error:
    metadata = DIAGNOSTIC_LAUNCH_METADATA or new_diagnostic_metadata(RUN_ROOT)
    write_json(metadata / "notebook_error.json", {"error": type(error).__name__ + ": " + str(error),
               "traceback": traceback.format_exc()})
    print({"diagnostic_stopped": type(error).__name__ + ": " + str(error),
           "original_retained": str(RUN_ROOT / RESUME_NAME), "metadata": str(metadata)})
finally:
    REPORT, RESULTS_ZIP = package_diagnostic(RUN_ROOT)
'''),
        cell("package_info", "markdown", "## 汇总与回传\n\n"
             "本格只读取和打包现有证据，不调用模型，可在启动失败后单独运行。"
             "ZIP 包含诊断、source05 比较输入、三臂 trace、源码、环境和启动日志；"
             "原批次保留在 Drive，仅引用其路径，不重复打包整个原目录或外部 ZIP。"
             "原累计、新增和合并累计调用分开报告。\n"),
        cell("package", "code", '''REPORT, RESULTS_ZIP = package_diagnostic(RUN_ROOT)
print(json.dumps({"counts_terminal": REPORT["counts_terminal"], "new_counts_status": REPORT["new_counts_status"],
                  "prior_actual_calls": REPORT["prior_actual_calls"], "actual_calls": REPORT["actual_calls"],
                  "cumulative_actual_calls": REPORT["cumulative_actual_calls"],
                  "classification": REPORT["classification"], "results_zip": str(RESULTS_ZIP)},
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
