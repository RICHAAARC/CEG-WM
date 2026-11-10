"""Build the directly runnable P2 Colab delivery with its own source snapshot.

No unpublished branch URL, source upload, SHA gate, or separate release needed.
Regenerate after any change in the included research implementation.
"""
from __future__ import annotations

import base64
import gzip
import io
from pathlib import Path
import tarfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/stage_protection_v2_colab.ipynb"


def source_files():
    # Include the research package, not .codex/governance, tests or results.
    return sorted([*ROOT.glob("src/cegwm/**/*.py"),
        ROOT / "experiments/run_stage_protection_v2.py",
        ROOT / "experiments/run_stage_protection_v2_resume.py",
        ROOT / "experiments/run_stage_protection_v2_compat_resume.py",
        ROOT / "experiments/run_reconstruction_channel_v1.py",
        ROOT / "configs/stage_protection_v2_colab.json",
        ROOT / "requirements-reconstruction-channel-colab.txt"])


def build(*, continuation=False, compatibility=False):
    continuation = continuation or compatibility
    output = ROOT / "notebooks/stage_protection_v2_resume_colab.ipynb" if continuation else OUTPUT
    if compatibility:
        output = ROOT / "notebooks/stage_protection_v2_compat_resume_colab.ipynb"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for path in source_files():
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(ROOT)))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    payload = base64.b64encode(gzip.compress(archive.getvalue(), mtime=0)).decode("ascii")
    cells = [nb.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')")]
    cells += [nb.v4.new_markdown_cell("""# CEG-WM 跨生成阶段证据保护 V2

本 notebook 内含固定源码，**Run all 直接执行已审定 P2**；无需上传源码、修改文件或勾选运行开关。
参数、输出目录与 GPU 元数据已预设；Colab 实际分配的 GPU 型号和可用性无法保证。
这是 2 个内容 × 独立 bit0/bit1 分支，合计 4 试次。CPU/fake 检查不代表真实实验成功。

early 在 step8 后，late 在 step18 后；reader、目标 cell、种子、位置与预算固定。
A/B/C 的任务验收统一使用 clean/design 固定 inner 与预算；C 私有保护区间单列诊断。
所有 proposal 与选择保存后才访问 holdout。P3/P4 不会自动执行。
11 步 early 反传的 GPU 显存、主机 RAM 与耗时 **未知**，没有 L4 可运行保证。
内存优化减少 suffix 的无用文本组件和稳定权重副本，但不保证解决 11 步激活累积。
本修订不授权重跑已 STARTED/FAILED 的操作；同目录仍保留其失败/中断状态，不自动重试。
""")]
    cells += [nb.v4.new_code_cell("""# 固定输出目录；再次 Run all 在同一目录恢复。
OUTPUT = '/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01'
""")]
    cells += [nb.v4.new_markdown_cell("""## 准备独立源码
源码从 notebook 内的固定快照解包。不会修改旧一次补偿或 LF/HF 结果。
同一输出目录只使用一个活动 VM；不要并行启动第二份 notebook。
中断后再次 Run all 复用已完成步骤，保留 STARTED/FAILED；不会自动重试这些步骤。
""")]
    cells += [nb.v4.new_code_cell("""import base64, gzip, io, os, sys, tarfile
from pathlib import Path
WORK = Path(os.environ.get('CEG_STAGE_PROTECTION_WORKDIR', '/content/ceg_stage_protection_v2'))
WORK.mkdir(parents=True, exist_ok=True)
BUNDLE = """ + repr(payload) + """
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(BUNDLE))), mode='r:') as archive:
    archive.extractall(WORK, filter='data')
sys.path[:0] = [str(WORK / 'src'), str(WORK)]
print('独立源码:', WORK)
""", metadata={"cellView": "form"})]
    cells += [nb.v4.new_markdown_cell("""## 冻结计划与成本
最终候选最多 16 个，不能当总模型调用数。SD3 最多 40 次 forward / 258 步；
SDXL 最多 56 次 forward / 336 步（44 ordinary + 12 proxy）；RGB/latent VJP 各 16；
reader encode 最多 102（70 结果 ordinary + 16 辅助 ordinary + 16 proxy）。
这些是不同层次的计数，不能相加为独立模型次数。实际记录包括 attempted、returned、失败和复用。
""")]
    cells += [nb.v4.new_code_cell("""import json
config = json.loads((WORK / 'configs/stage_protection_v2_colab.json').read_text(encoding='utf-8-sig'))
print(json.dumps(config, ensure_ascii=False, indent=2))
print('本次执行：固定 P2；不会自动进入 P3/P4。')
""")]
    cells += [nb.v4.new_markdown_cell("""## 准备环境并复用已有认证
沿用已固定的 diffusers/transformers 依赖和模型 revisions。保留 Colab 的 CUDA torch。
自动检查环境 token、已有 Hugging Face 登录缓存和 Colab secret `HF_TOKEN`。
仅在现有凭据实际无法访问固定模型时，才显示必要的权限说明与隐藏 token 输入；已有访问权限时不会强制登录。
依赖安装、模型下载、硬件资源与真实精度兼容性仍需在实际 Colab 验证。
""")]
    cells += [nb.v4.new_code_cell("""from cegwm.runtime.stage_protection_colab import ensure_model_access, record_setup_errors, run_logged
with record_setup_errors(OUTPUT):
    import torch
    assert sys.version_info >= (3, 11), '需要 Python 3.11+'
    assert tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 6), '需要 Colab torch >= 2.6'
    assert torch.cuda.is_available(), 'Colab 当前未提供 CUDA GPU；本次未启动 P2。'
    install_code = run_logged([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                              str(WORK / 'requirements-reconstruction-channel-colab.txt')],
                             cwd=WORK, output=OUTPUT, log_name='setup.log')
    if install_code != 0:
        raise RuntimeError('依赖安装失败；完整输出已保存到 ' + str(Path(OUTPUT) / 'setup.log'))
    access = ensure_model_access(config)
    print('模型访问已确认；认证来源:', access['credential_source'])
""")]
    cells += [nb.v4.new_markdown_cell("""## 运行或恢复固定首轮
结果直接写入上述 Drive 目录。每个操作开始前写 STARTED；发生中断不会覆盖为成功。
失败保持四试次分母；不扩来源、换 seed、换 cell、加预算或改变门槛。
runner 的 stdout/stderr 直接追加到 `runner.log`，环境准备错误保存在 `setup.log`；页面关闭不会删除已写入的日志。
`runner_resources.jsonl` 尽力记录父进程可见的 cgroup 内存事件、限制、峰值及退出差值；
`runner.log` 的 SOURCE_MEMORY 行记录长链步边界 RSS 与累计 CPU 保存字节。
缺失数据为 unknown/null，遥测写失败不阻止运行；cgroup 是共享范围，不能单凭 -9 或计数变化认定本进程 OOM。
""")]
    cells += [nb.v4.new_code_cell("""environment = os.environ.copy()
environment['PYTHONPATH'] = os.pathsep.join([str(WORK / 'src'), str(WORK)])
returncode = run_logged([sys.executable, '-u', '-m', 'experiments.run_stage_protection_v2',
                         '--execute', '--output', OUTPUT], cwd=WORK, env=environment, output=OUTPUT)
print('进程退出码:', returncode, '持久日志:', str(Path(OUTPUT) / 'runner.log'))
if returncode not in (0, 2):
    print('进程未正常完成。下方如有结果，是目录中最后保存的结果；请结合 runner.log 查看本次错误。')
""")]
    cells += [nb.v4.new_markdown_cell("""## 查看全部试次、原始 proposal 与共同回退
共同三通道指标是 clean/design 固定 cell 任务及 holdout 盲读目标 bit；预算另列。
另列 holdout 相对 design 固定 cell 的位置诊断，避免把不同同 bit cell 混为固定 cell 成功。
准出按内容成组：两个内容均需主动 early 翻转、A 干扰、非零 C 保护和严格 C>B 改进，
另一 bit 不得出现 B 成功 C 失败。质量只报告；有明显质量代价时需审查取舍，不能宣称已隔离机制收益。
即使出现操作性信号，P3 也需来源审计会话独立审阅，不自动扩跑。
""")]
    cells += [nb.v4.new_code_cell("""if (Path(OUTPUT) / 'summary.json').exists():
    import pandas as pd
    from IPython.display import display
    summary = json.loads((Path(OUTPUT) / 'summary.json').read_text())
    rows = []
    for trial in summary['trials']:
        for arm in ('A', 'B', 'C'):
            proposal = trial.get('arms', {}).get(arm, {})
            rows.append({'source': trial['source_id'], 'bit': trial['bit'], 'arm': arm,
                'status': proposal.get('status', 'incomplete'), 'selected': proposal.get('selected'),
                'raw_success': proposal.get('raw_three_channel_success'),
                'returned_success': proposal.get('returned_three_channel_success'),
                'protection_realized': proposal.get('protection_realized') if arm == 'C' else None,
                'C_B_actual_different': trial.get('C_B_actual_different') if arm == 'C' else None,
                'protection_active': trial.get('protection_constraint_active') if arm == 'C' else None,
                'early_relative': trial.get('early_relative'), 'late_relative': proposal.get('late_relative'),
                'RGB_MSE': proposal.get('quality', {}).get('mse'),
                'design_margin_change': proposal.get('design_margin_change'),
                'errors': '; '.join(trial.get('errors', []))})
    display(pd.DataFrame(rows))
    print(json.dumps(summary['verdict'], ensure_ascii=False, indent=2))
    display(pd.DataFrame([summary['physical_counts']]).T.rename(columns={0: 'observed_count'}))
    print('工程完成:', summary['engineering_complete'], '完整记录:', OUTPUT)
else:
    print('尚无 P2 汇总；检查 runner.log、setup.log 与已保存的操作记录，不代表实验通过。')
""")]
    cells += [nb.v4.new_code_cell("""if (Path(OUTPUT) / 'evaluated.pt').exists():
    import torch
    import matplotlib.pyplot as plt
    evaluated = torch.load(Path(OUTPUT) / 'evaluated.pt', map_location='cpu', weights_only=False)
    for trial in evaluated['trials']:
        if 'E' not in trial:
            print(trial['source_id'], trial['bit'], trial.get('errors'))
            continue
        panels = [('E', trial['E']['image'])]
        for arm in ('A', 'B', 'C'):
            proposal = trial['arms'].get(arm, {})
            observation = proposal.get('observation') or proposal.get('render')
            if observation:
                panels.append((arm + ' raw; return=' + proposal['selected'], observation['image']))
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
        for ax, (label, image) in zip(axes[0], panels):
            ax.imshow(image); ax.set_title(label); ax.axis('off')
        fig.suptitle(trial['source_id'] + ' / bit' + str(trial['bit']))
        plt.tight_layout(); plt.show()
""")]
    if continuation:
        cells[1].source = """# CEG-WM V2：从已有 checkpoints 继续

直接 **Run all**，无需上传、开关或手填路径。沿用 `attempt01` 的固定 P2。
仅对 `source06_fur_knit/bit1/early1/latent_vjp` 的原 STARTED 中断授权 **一次**新尝试。
其余已完成操作直接读取，随后继续 source06 剩余工作、source07 与冻结后的 holdout。
reader、基点、种子、预算、阈值、四试次分母与 holdout 冻结次序均不变，不自动执行 P3/P4。

runner 在加载模型前逐个校验必要 checkpoints；缺失或损坏即停止，不会重生成 source06。
原始状态、结果、计数不改写。新结果位于原目录的固定 `repairs/source06_bit1_early1_latent_vjp_attempt01/` 子目录。
新尝试 RETURNED 后再次 Run all 直接复用；新尝试 STARTED/FAILED 后不会再次自动重试。
内存优化未经真实 GPU 验证，11 步反传的 RAM、显存与耗时仍未知；CPU 检查不是科学结果。
"""
        cells[3].source = """## 准备内含源码
首格只挂载 Drive；源码随 notebook 提供。只使用一个活动 VM，不要同时运行原首轮 notebook。
续跑检查固定必要完成集并逐个释放加载值；原始中断记录保留，新尝试在物理计算前写入 STARTED。
"""
        cells[5].source = """## 固定方法与累计成本
原 P2 的 40 次 SD3 forward / 258 步等上限描述不间断原计划，**不再作为累计续跑成本上限**。
`continuation_counts.json` 分列原始消耗、本次新消耗、目标重试、之后续跑和累计计数。
旧中断已消耗的 attempted/returned 差额保留；不会清零或把重试当免费。
"""
        cells[6].source += "\nRESULTS = Path(OUTPUT) / 'repairs/source06_bit1_early1_latent_vjp_attempt01'\nprint('续跑结果:', RESULTS)\n"
        cells[9].source = cells[9].source.replace("## 运行或恢复固定首轮", "## 运行固定的一次修复与续跑")
        cells[10].source = cells[10].source.replace("'experiments.run_stage_protection_v2'", "'experiments.run_stage_protection_v2_resume'")
        for index in (12, 13):
            cells[index].source = cells[index].source.replace("Path(OUTPUT)", "RESULTS")
        cells[12].source += """
if (RESULTS / 'continuation_counts.json').exists():
    print('最近一次派生计数报告（SIGKILL 时可能滞后）：')
    print(json.dumps(json.loads((RESULTS / 'continuation_counts.json').read_text()), ensure_ascii=False, indent=2))
for label, path in (
    ('当前持久累计计数', RESULTS / 'physical_counts.json'),
    ('当前一次重试状态', RESULTS / 'operations/source06_fur_knit/bit1/early1/latent_vjp/status.json'),
):
    if path.exists():
        print(label, json.dumps(json.loads(path.read_text()), ensure_ascii=False, indent=2))
"""
    if compatibility:
        cells[1].source = """# CEG-WM V2：兼容性修复后继续 checkpoints

直接 **Run all**，无需上传、开关或手填路径。仅继续原固定 P2 的同一个中断目标。
本入口只接受首次 repair 的 `source06_fur_knit/bit1/early1/latent_vjp` 为 FAILED，且错误精确为
`source_suffix_vjp: ImportError: cannot import name 'is_non_overlapping_and_dense' from 'torch._prims_common'`。
首次 repair 增量必须只有 source pipeline 加载 attempted/returned 各 1，其他计算全为 0，且没有后来操作或冻结结果。
runner 在构造 provider / 加载模型前完成这些检查及原必要 checkpoints 的逐项校验；不匹配即停止。

保留原中断和首次兼容性失败的状态、partial、计数；从首次 repair 的最终累计计数继续，不清零加载成本。
新结果在原 `attempt01` 的固定 `repairs/source06_bit1_early1_latent_vjp_compat_attempt01/` 子目录。
此新尝试只执行一次；RETURNED 复用，STARTED/FAILED 不再自动重试。
成功后继续剩余固定 P2，已完成结果、基点、种子、reader、预算与 holdout 冻结次序保持不变，不自动 P3/P4。
私有接口依赖已移除；CPU 检查不代表真实 GPU 内存或科学结果已通过。
"""
        cells[3].source = """## 准备内含源码
首格仅挂载 Drive。源码已内含；使用一个活动 VM，不并行运行其他首轮或续跑 notebook。
精确的导入故障及零计算边界是本次机械修复的条件；没有一般失败重试开关。
"""
        cells[5].source += "\n首次 repair 的加载消耗也单列保留；新累计从该次最终 counts 开始。\n"
        cells[6].source = cells[6].source.replace("source06_bit1_early1_latent_vjp_attempt01", "source06_bit1_early1_latent_vjp_compat_attempt01")
        cells[10].source = cells[10].source.replace("'experiments.run_stage_protection_v2_resume'", "'experiments.run_stage_protection_v2_compat_resume'")
    mode = "embedded_source_run_all_P2_compatibility_repair" if compatibility else (
        "embedded_source_run_all_P2_one_repair" if continuation else "embedded_source_run_all_P2")
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": output.name}, "accelerator": "GPU",
        "stage_protection_delivery": {"mode": mode, "scientific_results": False}})
    nb.validate(notebook)
    nb.write(notebook, output)
    print(f"Wrote {output}; {len(source_files())} source files; no model execution")


if __name__ == "__main__":
    build()
