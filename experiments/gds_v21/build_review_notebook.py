"""Build fresh-runtime CPU review notebook. Data comes only from persisted Drive."""
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
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('geometry.py','review_g1.py'):
            info=zipfile.ZipInfo(name,date_time=(2026,11,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/name).read_bytes())
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest();encoded=base64.b64encode(payload).decode()
    nb=nbf.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(name='GDS_v2_1_G1_ReadOnly_Review.ipynb')))
    nb.cells=[nbf.v4.new_markdown_cell('''# G1 收尾审查：只读取 Google Drive 已有结果

**旧 runtime 断开没有影响。可使用全新的 CPU runtime，无需 GPU、旧 `/content` 文件或旧 Python 变量。**

输入固定为 Drive 中 `CEG-WM/gds-v21/runs/g1-v21-20261128-8sources-48attempts/`：报告、CSV、48 张图像、100 步 Token 轨迹、逐步日志及冻结几何资产。缺文件或一致性核验失败就停止，不会补跑。

不加载 VQ/扩散模型、不训练、不重新生成图像。仅用保存的 Token 做数值复核、读取既有图像制作审查页。原 G1 文件只读；派生表、图和 zip 写到 **Drive 独立的 `gds-v21/reviews/`** 目录，断开后仍保留。

输出覆盖：①落盘一致性；②匿名画质审查页；③t10/t0 保留率，区分重复事件与唯一位置；④70 个冻结几何竞争对的失败与晚期变化。画质需要人工看图评分；不会自动宣布 G2 通过。''')]
    nb.cells.append(nbf.v4.new_code_cell(f'''from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, io, json, os, subprocess, sys, uuid, zipfile
from google.colab import drive
drive.mount('/content/drive')
base = Path('/content/drive/MyDrive/CEG-WM/gds-v21')
source_dir = base / 'runs/g1-v21-20261128-8sources-48attempts'
assert (source_dir / 'report.json').is_file(), f'找不到 Drive 落盘报告：{{source_dir}}'
review_root = base / 'reviews'
review_root.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
output_dir = review_root / ('g1-readonly-' + stamp)
log_path = review_root / ('g1-readonly-' + stamp + '.log')
payload = base64.b64decode('{encoded}')
assert hashlib.sha256(payload).hexdigest() == '{sha}'
task_dir = Path('/content/g1-review-tools-{sha[:12]}')
task_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        assert Path(member.filename).name == member.filename
        (task_dir / member.filename).write_bytes(archive.read(member))
print('只读输入：', source_dir)
print('独立持久化输出：', output_dir)
print('日志：', log_path)
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 从 Drive 重读并核验

逐一检查 48 份结果：文件完整、100 步齐全、Token 与日志/指标/CSV 一致，重新计算的几何资产与原 hash 一致。核验后生成 8 页匿名画质图、保留率表和失败竞争对表；所有已读取的原始文件在分析前后比较 SHA256。

通常只需 CPU 和 Colab 预装的 NumPy/Pandas/Matplotlib/Pillow；本单元不安装生成模型库。若失败，回传下面打印的 traceback 和 Drive 日志。'''))
    nb.cells.append(nbf.v4.new_code_cell('''env = os.environ.copy()
env['MPLBACKEND'] = 'Agg'
env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, str(task_dir / 'review_g1.py'),
           '--source', str(source_dir), '--output', str(output_dir)]
with log_path.open('w') as log:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    try:
        for line in proc.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        returncode = proc.wait()
    except BaseException:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        raise
if returncode:
    raise RuntimeError(f'分析未完成。原始结果不补跑；检查日志：{log_path}')
review = json.loads((output_dir / 'review.json').read_text())
print(json.dumps({k: review[k] for k in ['completed_units', 'verified_inputs',
    'original_files_unchanged', 'human_quality_review', 'G2_decision']}, ensure_ascii=False, indent=2))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 画质：先看匿名页

每页 6 张既有图像，显示匿名 ID 和提示词，不显示方法。请检查明显结构缺陷、过强纹理/色偏、提示符合度；不要用 Token 命中率代替画质。

`quality_review_copy.csv` 是独立评价表副本，不覆盖原表。两位评价者应分别填写 1–5 分画质、1–5 分提示符合度、0/1 严重缺陷；有条件时独立评价，不先共享判断。仅一人检查或已看过方法标签时应说明，不能声称完成严格双人盲评。'''))
    nb.cells.append(nbf.v4.new_code_cell('''from IPython.display import display, Image as DisplayImage
for page in sorted(output_dir.glob('quality_blind_*.png')):
    print(page.name)
    display(DisplayImage(filename=str(page)))
print('评价表副本：', output_dir / 'quality_review_copy.csv')
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### t10/t0：参考是否保留下来

`unique_group_retention` 以每个位置最后一次成功写入的组符号为参考，每位置只计一次；`event_group_retention` 按成功事件计数，同一位置可能重复。组仍正确但 Token 已变，不等于参考丢失；端点正确也不能证明中间从未消失。N 没有写入事件，其保留率为空值。

下面的平均数以源为单位；不要把逐步事件数当作独立样本量。'''))
    nb.cells.append(nbf.v4.new_code_cell('''import pandas as pd
arms = pd.read_csv(output_dir / 'arm_summary.csv')
cols = ['arm', 'sources', 'final_address_hit_fraction', 'min_margin_t0',
        'unique_success_positions', 'unique_group_retention_t10', 'unique_group_retention_t0',
        'unique_exact_token_retention_t0', 'event_group_retention_t0']
display(arms[cols])
display(DisplayImage(filename=str(output_dir / 'retention.png')))
display(DisplayImage(filename=str(output_dir / 'margin_by_source.png')))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 失败的几何竞争对

P 每个源最终最差的**同一个**竞争对，会分别列出 t11、t10、t0 间隔。正值代表该固定比较中真几何得分更高；≤0 表示错误几何胜出或打平。这是理想 Token 符号链，不是 RGB 同步成功率。

`positive_at_11_nonpositive_final` 表示写入结束时该对为正、最终变为非正；`already_nonpositive_at_11_and_10` 表示在这两个端点就未解决。中间恢复后再失败另列。最终最差对属于事后诊断，不能当作事先独立测试集。

参数顺序是 `(角度°, 尺度, tx像素, ty像素)`，数学 y 轴向上。输出有效支持比例不等同裁剪后保留的源图面积。'''))
    nb.cells.append(nbf.v4.new_code_cell('''worst = pd.read_csv(output_dir / 'P_worst_pairs.csv')
for prefix in ['r', 'h']:
    worst[prefix + '_transform'] = worst.apply(lambda row: tuple(round(float(row[prefix + '_' + p]), 4)
        for p in ['angle', 'scale', 'tx', 'ty']), axis=1)
display(worst[['source_id', 'pair_id', 'r_transform', 'h_transform', 'relative_type',
               'margin_t11', 'margin_t10', 'margin_t0', 'failure_stage']])
frequency = pd.read_csv(output_dir / 'failure_pair_frequency.csv')
display(frequency[(frequency.arm == 'P') & (frequency.nonpositive_sources > 0)].head(15))
display(DisplayImage(filename=str(output_dir / 'P_pair_heatmap.png')))
'''))
    nb.cells.append(nbf.v4.new_markdown_cell('''### 持久化结果与回传

原始 G1 目录不会改变。所有派生结果及 zip 已放在 Drive 的独立审查目录。请回传 `*_review.zip`（包含图页、表和核验摘要），或先回传 `arm_summary.csv`、`P_worst_pairs.csv` 和画质观察。包内匿名图像页是原有 48 张生成图的汇集，请按需要分享。

**到此停止，不启动 G2。** 要结合画质、端点保留和失败类型再判断；本 notebook 没有自动设置新的“过关阈值”。'''))
    nb.cells.append(nbf.v4.new_code_cell('''archive_path = Path(str(output_dir) + '_review.zip')
print('审查目录（Drive 持久保存）：', output_dir)
print('回传压缩包（Drive 持久保存）：', archive_path)
print('原始 G1 输入未变：', review['original_files_unchanged'])
print('G2：尚未启动，等待收尾审查判断。')
# 如需下载到本机，可取消下一行注释；不会修改原始 G1 文件。
# from google.colab import files; files.download(str(archive_path))
'''))
    nbf.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code': ast.parse(c.source,filename=f'cell-{i}')
    out=root.parents[1]/'notebooks/GDS_v2_1_G1_ReadOnly_Review.ipynb';nbf.write(nb,out)
    print(json.dumps(dict(notebook=str(out),payload_sha256=sha,inputs='Google Drive only',
                         source_mutations=False,model_loading=False,real_G1_files_accessed_locally=False),indent=2))


if __name__=='__main__':build()
