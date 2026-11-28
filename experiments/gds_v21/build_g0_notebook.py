"""Build a self-contained G0-only Colab. Never load a checkpoint during the build."""
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
    names=['core.py','geometry.py','compat.py','observe.py','test_g0.py','g0_cpu.py','g0_gpu.py','g0_config.json','README.txt']
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,11,28,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/name).read_bytes())
    payload=archive.getvalue()
    encoded=base64.b64encode(payload).decode()
    digest=hashlib.sha256(payload).hexdigest()
    nb=nbf.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name='GDS_v2_1_G0_Check.ipynb'),accelerator='GPU'))
    nb.cells=[nbf.v4.new_markdown_cell('''# GDS v2.1：G0 工程门槛检查

**本 notebook 只运行 G0，不运行 G1，不训练读取器。**

在 Colab 选择「运行时 → 更改运行时类型 → GPU」，然后「全部运行」。使用当前 Colab 环境，安装固定版本的 Diffusers/Transformers；不安装旧 CEG 环境、不替换 torch/CUDA。

先检查几何与新核，再用同一 GPU/process 对一个诊断源原生生成 **2 次**（未加水印）：比较无核观察与带核观察的 100 步 Token、RNG 和图像，核验真实概率及冻结 t=60..11 的可交换窗口。这两次属于 G0，**不是 G1 的 48 次**。

成功条件为最后显示 `G0_REAL_MODEL: passed`；CPU passed 单独不够。日志或模型检查失败即停，不自动重试。完成后回传最后摘要及 `report.json`。确认完整 G0 通过后，再准备 **8 个新源 × 6 臂 = 48 次生成上限** 的 G1 notebook。

这是工程检查，不是水印可读性、鲁棒性或创新性结果。''')]
    nb.cells.append(nbf.v4.new_code_cell('''from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, io, json, os, subprocess, sys, uuid, zipfile
from google.colab import drive
drive.mount('/content/drive')
run_root = Path('/content/drive/MyDrive/CEG-WM/gds-v21/runs')
run_root.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
run_dir = run_root / ('g0-' + stamp)
run_dir.mkdir(exist_ok=False)
print('Output:', run_dir)
# Reuse existing model weights only; all new outcomes have a fresh output folder.
os.environ['HF_HOME'] = '/content/drive/MyDrive/CEG-WM/discrete-pair-v1/model-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

def run_logged(command, name, cwd=None):
    path = run_dir / (name + '.log')
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
        print('\nFAILED. Full error log:', path)
        raise RuntimeError(f'{name} failed (exit {code}); see printed traceback and {path}')

run_logged([sys.executable, '-m', 'pip', 'install', '--quiet',
            'diffusers==0.39.0', 'transformers==4.57.6',
            'accelerate', 'sentencepiece', 'safetensors', 'pytest'], 'install')
'''.replace("print('\nFAILED", "print('\\nFAILED")))
    nb.cells.append(nbf.v4.new_code_cell(f'''# Self-contained, SHA256-checked payload; no mutable GitHub source downloads.
payload = base64.b64decode('{encoded}')
expected_sha256 = '{digest}'
assert hashlib.sha256(payload).hexdigest() == expected_sha256
task_dir = Path('/content/gds-v21-g0-' + expected_sha256[:12])
task_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert Path(member.filename).name == member.filename
        target = task_dir / member.filename
        content = archive.read(member)
        if target.exists():
            assert target.read_bytes() == content, 'Existing payload changed; use a fresh runtime'
        else:
            target.write_bytes(content)
(run_dir / 'payload.sha256').write_text(expected_sha256)
print('Payload:', expected_sha256)
run_logged([sys.executable, 'g0_cpu.py', '--output', str(run_dir / 'cpu')], 'g0_cpu', task_dir)
cpu_report = json.loads((run_dir / 'cpu/report.json').read_text())
assert cpu_report['status'] == 'passed'
print('CPU passed. Full G0 still requires the next CUDA check.')
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 同 runtime 真实模型检查

下面的单元最多执行两次原生生成，不做写入。首次运行可能需要下载模型。若中断或失败，保留输出目录并回传错误；不要把失败记录删掉后补成成功。

每个生成尝试限时 900 秒；模型下载时间另计。当前 notebook 没有 G1 或读取器训练入口。'''))
    nb.cells.append(nbf.v4.new_code_cell('''assert cpu_report['status'] == 'passed'
try:
    run_logged([sys.executable, 'g0_gpu.py', '--cpu-report', str(run_dir / 'cpu/report.json'),
                '--output', str(run_dir / 'gpu')], 'g0_gpu', task_dir)
finally:
    report_path = run_dir / 'gpu/report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        summary = {key: report.get(key) for key in ['stage', 'status', 'decision', 'error',
            'gpu', 'versions', 'checks', 'observed_native_opportunity_timesteps',
            'attempts', 'wall_seconds', 'G1_started', 'reader_training_started']}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Report:', report_path)
    print('All records:', run_dir)
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''**到此停止。** 请回传上个单元的完整摘要；有失败时附 `g0_gpu.log` / `gpu/failure.txt`。只有完整 G0 通过后，才准备并交付 G1 链接。G1 固定 8 个新源、48 次生成；不自动进入训练或扩大实验。'''))
    nbf.validate(nb)
    for index,cell in enumerate(nb.cells):
        if cell.cell_type=='code':
            ast.parse(cell.source,filename=f'cell-{index}')
    destination=root.parents[1]/'notebooks'/'GDS_v2_1_G0_Check.ipynb'
    destination.parent.mkdir(exist_ok=True)
    nbf.write(nb,destination)
    print(json.dumps(dict(notebook=str(destination),payload_sha256=digest,
        payload_files=names,code_cells=sum(c.cell_type=='code' for c in nb.cells),
        runtime_execution='not run: CUDA/Colab unavailable locally'),indent=2))
    return destination


if __name__=='__main__':
    build()
