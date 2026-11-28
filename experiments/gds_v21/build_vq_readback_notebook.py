"""Self-contained, fixed-scope Colab readback notebook, code-only payload."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
import nbformat as nbf


def build():
    root=Path(__file__).resolve().parent
    names=('geometry.py','g2a_reader.py','g2a_config.json','vq_readback.py',
           'vq_readback_config.json','VQ_READBACK_README.txt')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,11,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/name).read_bytes())
    payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest();encoded=base64.b64encode(payload).decode()
    nb=nbf.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),accelerator='GPU',colab=dict(name='GDS_v2_1_Frozen_VQ_Readback.ipynb')))
    nb.cells=[nbf.v4.new_markdown_cell('''# 冻结原生 VQ 重编码：只读已有 48 张 G1 图

选择 **GPU runtime**，从上到下运行。可用全新的 Colab 环境，所有输入从 Google Drive 重读。

**不训练、不生成、不做攻击。每图仅一次 VQ 编码，最多 48 次。** 不运行解码器或扩散去噪器。使用原固定 checkpoint 的 encoder、quant_conv 和量化码本，拒绝缺失/随机补齐权重。

本次验证：RGB 能否恢复原 Token 与原随机16组；与已完成 G2-A 的保存结果比较；仅对 clean 读出场计算原有恒等几何竞争分数。原 Token 只用于离线评分，不参与选择读出。

输出与日志持续落盘；任务含读取/模型加载最多30分钟，超限保留记录停止。正常完成也不自动启动几何攻击或训练。''')]
    nb.cells.append(nbf.v4.new_markdown_cell('''### 1. 挂载 Drive 并检查输入

需要原 G1 目录里的图像与 Token 轨迹，以及 G2-A 的报告、指标和保存概率。`review.zip` 路径本身不能代替这些原件。本 notebook 不依赖旧 `/content` 或旧变量。

当前 Colab torch/CUDA 保留；安装与原生成代码一致的 diffusers/transformers，仅加载 VQ 组件。原有 Drive 模型缓存可复用。'''))
    nb.cells.append(nbf.v4.new_code_cell('''from pathlib import Path
import os, sys, json, subprocess, base64, hashlib, io, zipfile
from google.colab import drive
drive.mount('/content/drive')
import torch
assert torch.cuda.is_available(), '请切换到 GPU runtime。'
base = Path('/content/drive/MyDrive/CEG-WM/gds-v21')
source_dir = base/'runs/g1-v21-20261128-8sources-48attempts'
reference_dir = base/'runs/g2a-clean-existing-g1-v1'
output_dir = base/'runs/vq-readback-existing-g1-v1'
assert not output_dir.exists(), f'本次输出已存在：{output_dir}。请先回传记录，不删除目录重跑。'
for parent, names in [(source_dir,['report.json']), (reference_dir,['report.json','input_hashes.json','metrics.csv','group_probabilities_64.npz'])]:
    for name in names: assert (parent/name).is_file(), f'缺少 Drive 原件：{parent/name}'
log_dir = base/'delivery-logs'; log_dir.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']='/content/drive/MyDrive/CEG-WM/discrete-pair-v1/model-cache'
os.environ['PYTHONUNBUFFERED']='1'; os.environ['MPLBACKEND']='Agg'
print('GPU:',torch.cuda.get_device_name(0), '| torch:',torch.__version__)
print('只读输入:',source_dir, reference_dir)
print('独立输出:',output_dir)

def run_logged(command, name):
    path=log_dir/(name+'.log')
    print('Running:',name,'| log:',path,flush=True)
    with path.open('x') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
        try:
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
        except BaseException:
            proc.send_signal(2)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    if code:raise RuntimeError(f'已停止，不补跑；回传日志：{path}')
    return path

run_logged([sys.executable,'-m','pip','install','--quiet','diffusers==0.39.0',
            'transformers==4.57.6','accelerate','safetensors'], 'vq-readback-v1-setup')
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 2. 解包固定验证程序

代码嵌入本 notebook 并校验 SHA256，不读取浮动分支。复用 G2-A 的数据检查与指标函数，不创建 U-Net，也不调用训练函数。
'''))
    nb.cells.append(nbf.v4.new_code_cell(f'''payload=base64.b64decode('{encoded}')
assert hashlib.sha256(payload).hexdigest()=='{sha}'
task_dir=Path('/content/vq-readback-tools-{sha[:12]}');task_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert Path(member.filename).name==member.filename
        (task_dir/member.filename).write_bytes(archive.read(member))
print(json.dumps(json.loads((task_dir/'vq_readback_config.json').read_text()),ensure_ascii=False,indent=2))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 3. 48 张原图逐一冻结回读

核验 G1 与 G2-A 记录一致后，每图仅一次编码。码距/原 Token 排名复用该次特征；top-k 是诊断，不能用真值从候选中选答案。量化直接使用官方函数，没有可调温度。

进度、每图索引、混淆及状态均保存到 Drive。缺文件、权重不完整、OOM 或超时均停止，失败和未运行图保留在48个固定名额内。断线后可从 Drive 找报告和日志；本版不自动重试/恢复。'''))
    nb.cells.append(nbf.v4.new_code_cell('''run_logged([sys.executable,str(task_dir/'vq_readback.py'),
            '--source',str(source_dir),'--reference',str(reference_dir),
            '--output',str(output_dir)], 'vq-readback-v1-run')
report=json.loads((output_dir/'report.json').read_text())
display_keys=['status','completed_units','encoder_forwards','decoder_forwards','optimizer_steps',
              'new_generations','attacks','wall_seconds','original_files_unchanged','decision']
print(json.dumps({k:report.get(k) for k in display_keys},ensure_ascii=False,indent=2))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 4. 留出可读性与混淆

主要看留出源 N/B0，P 及其他臂并列报告。原先的源划分保持不变；这里没有训练，沿用名称只是便于对照。两个留出源不是新的独立确认集。

`token_accuracy`：原索引精确恢复；`group_accuracy`：原随机组恢复；`cross_group_given_token_error`：读错索引时跨组的比例。精确Token正确必然组正确。硬读出的裁剪交叉熵不与 U-Net 软概率比较。
'''))
    nb.cells.append(nbf.v4.new_code_cell('''import pandas as pd
from IPython.display import display, Image as DisplayImage
token_table=pd.read_csv(output_dir/'token_summary.csv')
comparison=pd.read_csv(output_dir/'reader_comparison.csv')
controls=pd.read_csv(output_dir/'group_metrics.csv')
display(token_table[token_table.split.eq('heldout')][['unit','token_accuracy','group_accuracy',
    'cross_group_given_token_error','true_token_top4','true_token_top16','true_token_rank_median']])
display(comparison[comparison.split.eq('heldout')])
display(controls[controls.split.eq('heldout') & controls.arm.isin(['N','B0'])].pivot_table(
    index=['unit','method'],values=['accuracy','balanced_accuracy']))
display(DisplayImage(filename=str(output_dir/'heldout_readback.png')))
display(DisplayImage(filename=str(output_dir/'token_vs_group.png')))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 5. 仅 clean 参考场的几何竞争诊断

比较真实恒等位置与原来8个错误假设，不变换图像、不运行攻击、不修改评分归一化。
`ideal_tokens` 仅为诊断真值；`saved_unet` 来自旧 G2-A 保存概率，不再运行其网络；`native_vq` 是本次盲回读。

即使分类提高，若正确几何仍不能胜出，也不能宣布可进入攻击。此处不是完整盲几何搜索或固定误报率检测。'''))
    nb.cells.append(nbf.v4.new_code_cell('''geometry=pd.read_csv(output_dir/'clean_geometry.csv')
display(geometry[geometry.split.eq('heldout')].groupby(['unit','method']).margin.agg(['min','mean']))
display(DisplayImage(filename=str(output_dir/'clean_identity.png')))
print('回传压缩包（Drive 持久保存）：',str(output_dir)+'_review.zip')
print('到此停止，不自动追加训练、新图或攻击。')
'''))
    nbf.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':ast.parse(c.source,filename=f'cell-{i}')
    target=root.parents[1]/'notebooks/GDS_v2_1_Frozen_VQ_Readback.ipynb'
    nbf.write(nb,target)
    print(json.dumps(dict(notebook=str(target),payload_sha256=sha,real_model_executed=False),indent=2))


if __name__=='__main__':build()
