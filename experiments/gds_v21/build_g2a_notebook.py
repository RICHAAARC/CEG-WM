"""Self-contained Colab delivery. Payload has code only, no user data."""
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
    buffer=io.BytesIO()
    names=('geometry.py','g2a_reader.py','g2a_config.json','G2A_README.txt')
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,11,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/name).read_bytes())
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest()
    encoded=base64.b64encode(payload).decode()
    nb=nbf.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name='GDS_v2_1_G2A_Clean_RGB_Pilot.ipynb'),accelerator='GPU'))
    nb.cells=[nbf.v4.new_markdown_cell('''# G2-A：现有 G1 图像的 clean RGB 可读性先导

**选择 GPU runtime，然后从上到下运行。旧 `/content` 已丢失不影响：全部输入来自 Google Drive。**

本次只验证 RGB 能否读出实际 Token 的 16 组标签。零次新生成，不加载扩散模型；仅训练一个小型 U-Net。画质继续记录，未完成的人工评分不阻塞本次运行，也不会写成通过。

**预算固定：1 次训练，最多 2,000 步、batch 4、总墙钟 90 分钟。** 8 个已有源按固定 hash 分为 4 训练 / 2 开发 / 2 留出；训练只用训练源的 N/B0，共 8 张图，开发监测用 4 张。只使用最后一步模型，不挑最佳 checkpoint。训练权重、日志及结果持续保存到 Drive；再次运行会拒绝覆盖同一输出目录。

这是完整 G2 之前的小数据先导，**不是原纸面 256+64 源实验**。两留出源只隔离于训练，并非未被研究者看过的新确认集。尚不执行 JPEG、几何恢复、主水印或扩量。''')]
    nb.cells.append(nbf.v4.new_markdown_cell('''### 1. 挂载 Drive、核对运行范围

不安装旧 CEG 环境、不替换 torch/CUDA；使用当前 Colab 的 torch、NumPy、Pillow、Pandas、Matplotlib，实际版本会落盘。缺少依赖或 GPU 就停止。输出目录固定，以防误点“全部运行”触发重复训练。'''))
    nb.cells.append(nbf.v4.new_code_cell(f'''from pathlib import Path
import base64, hashlib, io, json, os, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
import torch, numpy, PIL, pandas, matplotlib
assert torch.cuda.is_available(), '请切换到 GPU runtime；本 notebook 不在 CPU 上训练。'
base = Path('/content/drive/MyDrive/CEG-WM/gds-v21')
source_dir = base / 'runs/g1-v21-20261128-8sources-48attempts'
output_dir = base / 'runs/g2a-clean-existing-g1-v1'
assert (source_dir / 'report.json').is_file(), f'缺少 G1 原始落盘目录：{{source_dir}}。review.zip 不能代替原图和 Token 轨迹。'
assert not output_dir.exists(), f'本次尝试已存在：{{output_dir}}。请回传已有报告/日志，不删目录重跑。'
payload = base64.b64decode('{encoded}')
assert hashlib.sha256(payload).hexdigest() == '{sha}'
task_dir = Path('/content/g2a-tools-{sha[:12]}'); task_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert Path(member.filename).name == member.filename
        (task_dir / member.filename).write_bytes(archive.read(member))
sys.path.insert(0, str(task_dir))
from g2a_reader import source_split
config = json.loads((task_dir/'g2a_config.json').read_text())
split = source_split([f'g1-{{i:03}}' for i in range(8)], config)
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
print('源级划分：', json.dumps(split, ensure_ascii=False, indent=2))
print('训练预算：', config['max_steps'], '步；', config['max_run_seconds']/60, '分钟上限')
print('输入（只读）：', source_dir)
print('输出（Drive）：', output_dir)
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 2. 执行一次有限训练与留出评估

标签来自实际最终 Token，不是理想空间模板。所有同源方法都留在同一集合；P/B1/B2/P−E 不参加训练。每 100 步保存 `latest.pt`，每 200 步记录开发交叉熵，只监测、不选模型。

出现输入损坏、OOM、非有限值或预算耗尽就停止并保留记录，不自动缩模型、重试或追加训练。正常完成后在全部 48 图上输出分集合诊断，**只有 2 个留出源的 12 张图属于未参与拟合的评估**。

浏览器断开不保证后台继续执行；如果 runtime 被回收，Drive 留下的报告/检查点可供审查，本版不自动恢复训练。'''))
    nb.cells.append(nbf.v4.new_code_cell('''log_dir = base / 'delivery-logs'; log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / 'g2a-clean-existing-g1-v1.log'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['MPLBACKEND'] = 'Agg'
command = [sys.executable, str(task_dir/'g2a_reader.py'),
           '--source', str(source_dir), '--output', str(output_dir)]
print('Running: g2a_clean_reader | log:', log_path, flush=True)
with log_path.open('x') as log:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    try:
        for line in proc.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        code = proc.wait()
    except BaseException:
        proc.send_signal(2)
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        raise
report = json.loads((output_dir/'report.json').read_text()) if (output_dir/'report.json').exists() else {}
print(json.dumps(report, ensure_ascii=False, indent=2))
if code:
    raise RuntimeError(f'本次停止，不补跑。回传 Drive report.json / failure.txt / 日志：{log_path}')
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 3. 看留出可读性，而不只看训练拟合

`reader` 必须优于训练集组频先验、位置先验，以及使用另一源 RGB 输出、打乱空间输出的控制。N 单独报告，因为它没有目标地址偏置。`address` 是理想地址猜测，用于揭示只背空间模板的捷径。

balanced accuracy 为每张图实际出现类别的平均召回率；交叉熵越低越好。类别支持、混淆矩阵和逐图结果均保存。不要把 1,024 个 Token 或六个方法臂当成独立源。'''))
    nb.cells.append(nbf.v4.new_code_cell('''import pandas as pd
from IPython.display import display, Image as DisplayImage
table = pd.read_csv(output_dir/'metrics.csv')
heldout = table[table.split.eq('heldout')]
display(heldout[heldout.arm.isin(['N','B0'])].pivot_table(
    index=['source_id','arm','method'], values=['accuracy','balanced_accuracy','ce']))
display(heldout[heldout.method.eq('reader')][['source_id','arm','accuracy','balanced_accuracy','ce','present_classes']])
display(DisplayImage(filename=str(output_dir/'learning_curve.png')))
display(DisplayImage(filename=str(output_dir/'heldout_controls.png')))
display(pd.DataFrame(report['signal_flags']))
print('诊断：', report['decision'])
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 4. 到此停止并回传

预先定义的“小样本信号”标记要求四个留出 N/B0 图分别优于指定控制；它只是下一轮审查线索，不是统计显著性或发表门槛。训练好但留出差，或不优于捷径控制：本小数据预算尚未建立可读性，停止审查，不自动扩大。即使全部满足，也先审查读出场和证据，再决定是否进入真实 RGB 几何恢复。

本轮没有未见地址排列数据，故不能声称已完整排除布局依赖。失败也不能证明更充分数据下的读取器不可能。原 G1 结果与画质记录不变；本结果不证明画质可接受、盲同步成功或高鲁棒水印成立。'''))
    nb.cells.append(nbf.v4.new_code_cell('''print('回传压缩包（Drive）：', str(output_dir) + '_review.zip')
print('读取器检查点（Drive，单独保留）：', output_dir/'latest.pt')
print('原始输入未变：', report.get('original_files_unchanged'))
print('到此停止：不自动开始几何攻击、扩量或下一次训练。')
'''))
    nbf.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code': ast.parse(c.source, filename=f'cell-{i}')
    target=root.parents[1]/'notebooks/GDS_v2_1_G2A_Clean_RGB_Pilot.ipynb'
    nbf.write(nb,target)
    print(json.dumps(dict(notebook=str(target),payload_sha256=sha,
                         payload_files=names,real_training_executed=False),indent=2))


if __name__=='__main__':build()
