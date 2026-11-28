"""Self-contained limited G1 Colab, with an existing G0 report prerequisite."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import nbformat as nbf


def build():
    root=Path(__file__).resolve().parent
    names=['core.py','geometry.py','compat.py','observe.py','g0_cpu.py','g0_gpu.py','g0_config.json',
           'test_g0.py','test_g1.py','g1_sampling.py','g1_runner.py','g1_config.json','g1_g0_evidence.json','G1_README.txt']
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,11,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/name).read_bytes())
    payload=stream.getvalue();encoded=base64.b64encode(payload).decode();sha=hashlib.sha256(payload).hexdigest()
    nb=nbf.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name='GDS_v2_1_G1_8Sources_48Runs.ipynb'),accelerator='GPU'))
    nb.cells=[nbf.v4.new_markdown_cell('''# GDS v2.1：G1 最小验证（8 新源 / 48 次上限）

**G0 已由用户回传的 L4 日志确认通过。本 notebook 核对 Drive 原始报告后，才运行 G1。**

选择与 G0 一致的 **NVIDIA L4 GPU**，然后「全部运行」。沿用当前 Colab 环境及既有模型缓存，不需要旧 CEG 环境。包版本/GPU 类型与 G0 不符会停止；不会自动补跑 G0 或更换实验配置。

固定 8 个新 prompt/seed，每源 N、B0、B1、B2、P、P−E 各一次，共 **48 次生成尝试上限**。六臂都记录相同历史上的旁路选择；失败即停并保留分母，不自动重试、补图或扩大实验。

只看 Token 参考能否形成、动态账本是否改变决策及费用。**不训练读取器、不运行主水印或攻击检测、不据此报告 RGB 鲁棒性/FPR。** 最终图像与匿名质量检查表会保存，但人工质量评价尚待完成。

首次真实 G1 尚未执行；发布前已做源码/格式、CPU 调度器与采样接口测试，不能代替本次 GPU 验证。请保持 runtime 打开直到结束。''')]
    nb.cells.append(nbf.v4.new_code_cell('''from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, io, json, os, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
base = Path('/content/drive/MyDrive/CEG-WM/gds-v21')
g0_report = base / 'runs/g0-20261128T073448Z-bde6b4/gpu/report.json'
assert g0_report.exists(), f'缺少已通过的 G0 报告：{g0_report}'
assert json.loads(g0_report.read_text())['status'] == 'passed'
# Fixed path deliberately prevents an unnoticed second 48-run experiment.
run_dir = base / 'runs/g1-v21-20261128-8sources-48attempts'
assert not run_dir.exists(), f'已有 G1 目录，请先回传记录，不要删除后重跑：{run_dir}'
log_dir = base / 'delivery-logs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
log_dir.mkdir(parents=True, exist_ok=False)
os.environ['HF_HOME'] = '/content/drive/MyDrive/CEG-WM/discrete-pair-v1/model-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
print('G0:', g0_report)
print('G1 output:', run_dir)
print('Logs:', log_dir)

def run_logged(command, name, cwd=None):
    path = log_dir / (name + '.log')
    print('Running:', name, '| log:', path, flush=True)
    with path.open('w') as log:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
    if code:
        raise RuntimeError(f'{name} failed (exit {code}). 保留完整日志并回传：{path}')

run_logged([sys.executable, '-m', 'pip', 'install', '--quiet',
            'diffusers==0.39.0', 'transformers==4.57.6',
            'accelerate', 'sentencepiece', 'safetensors', 'pytest'], 'install')
'''))
    nb.cells.append(nbf.v4.new_code_cell(f'''# SHA256-checked frozen source payload. No branch-head code download.
payload = base64.b64decode('{encoded}')
payload_sha256 = '{sha}'
assert hashlib.sha256(payload).hexdigest() == payload_sha256
task_dir = Path('/content/gds-v21-g1-' + payload_sha256[:12])
task_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert Path(member.filename).name == member.filename
        data = archive.read(member)
        target = task_dir / member.filename
        if target.exists():
            assert target.read_bytes() == data, 'Payload changed; stop for review'
        else:
            target.write_bytes(data)
(log_dir / 'payload.sha256').write_text(payload_sha256)
run_logged([sys.executable, '-m', 'pytest', '-o', 'addopts=',
            'test_g0.py', 'test_g1.py', '-q'], 'cpu_tests', task_dir)
config = json.loads((task_dir / 'g1_config.json').read_text())
print('Frozen budget:', config['source_count'], 'new sources ×', len(config['arms']), '= 48 attempts')
for source in config['sources']:
    print(source['source_id'], source['seed'], source['prompt'])
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 执行 G1

48 个尝试在开始前登记。每张图 100 次生成器前向，旁路比较不增加模型前向；它的 CPU/GPU 计算和日志费用仍计入墙钟。保存每步位置选择、实际 Token、t10/t0 保留情况、图像与符号图。

每个尝试限时 900 秒，整轮含加载限时 4 小时；这是上限，不是预计完成时间。不要重复点击执行此单元。若中断或失败，固定目录会阻止从头重跑，请回传报告后再讨论恢复。'''))
    nb.cells.append(nbf.v4.new_code_cell('''assert not run_dir.exists(), f'G1 目录已存在，停止：{run_dir}'
try:
    run_logged([sys.executable, 'g1_runner.py', '--g0-report', str(g0_report),
                '--output', str(run_dir)], 'g1_generation', task_dir)
finally:
    report_path = run_dir / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(json.dumps({k: report.get(k) for k in ['stage', 'status', 'summary', 'stop_reason',
            'error', 'generation_attempts_started', 'wall_seconds', 'quality_review',
            'reader_training_started', 'G2_started']}, ensure_ascii=False, indent=2))
    print('Report:', report_path)
    print('Full log:', log_dir / 'g1_generation.log')
'''))
    nb.cells.append(nbf.v4.new_code_cell('''import csv
from IPython.display import display
import pandas as pd
summary_path = run_dir / 'summary.csv'
if summary_path.exists():
    table = pd.read_csv(summary_path)
    assert len(table) == 48
    display(table)
    print('匿名图像及空白双人评价表：', run_dir / 'blind_quality')
    print('符号图和实际生成图：各 source_arm 子目录。符号图不是 RGB 读取结果。')
print('到此停止；请回传上个单元的摘要和 summary.csv，不进入训练或扩大实验。')
'''))
    nbf.validate(nb)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code': ast.parse(cell.source,filename=f'cell-{i}')
    destination=root.parents[1]/'notebooks/GDS_v2_1_G1_8Sources_48Runs.ipynb'
    nbf.write(nb,destination)
    print(json.dumps(dict(notebook=str(destination),payload_sha256=sha,payload_files=names,
                         actual_G1_execution=False),indent=2))
    return destination


if __name__=='__main__': build()
