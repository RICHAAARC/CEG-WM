"""Self-contained Colab delivery of the frozen, gated candidate."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
template=json.loads((ROOT.parent/'CropResidual-V1/CEG_WM_crop_residual_v1.ipynb').read_text())
cells=[]
def cell(kind,text):
    item={'cell_type':kind,'id':f'cell-{len(cells):02d}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': item.update(execution_count=None,outputs=[])
    cells.append(item)

cell('markdown', '''# CEG-WM：固定跨通道互谱读法，先零位移门槛

**复用现有图像，不重新生成。先做零位移密钥区分；不通过即停止，不启动位移对照。**

沿用成功的 **Python 3.12 GPU 环境**与 Secrets `HF_TOKEN`、`CEG_WM_ROOT_KEY`，运行全部单元。
只读输入：`MyDrive/CEG-WM/crop-bottleneck-v1/attempt01/`。
本轮独立输出：`MyDrive/CEG-WM/crop-cross-spectrum-v1/attempt01/`。

只加载原版 VAE。B 为原共同掩膜的实部 HF 相关；C 为固定跨通道互谱，全部 120 个通道对等权。
每个频率除以全部 16 通道总能量；整体做复数余弦的实部，不取绝对值。不选频率、不选通道、不选几何、不训练。
只用原密钥与固定 16 个错钥，不继承联合检测阈值、不混入 LF。

本地工程验证不是 GPU 科学结果。本 notebook 尚未在你的保存图像上执行；运行后可能正确地停在零位移门槛。
''')
setup=''.join(template['cells'][1]['source'])
setup=setup.replace('crop-residual-v1','crop-cross-spectrum-v1').replace('cegwm-crop-residual-source','cegwm-crop-cross-spectrum-source')
setup=setup.replace("script = pathlib.Path('/content/run_crop_residual.py')", "script_dir = pathlib.Path('/content/cegwm-crop-cross-spectrum')\nscript_dir.mkdir(exist_ok=True)\nscript = script_dir / 'run_cross_spectrum.py'")
cell('code',setup)
cell('code',''.join(template['cells'][2]['source']).replace('cegwm-residual-cuda-constraints','cegwm-cross-spectrum-cuda-constraints'))
embedded='# Embedded fixed runner plus hash-bound support modules.\n'
for filename,location in [('run_residual.py',ROOT.parent/'CropResidual-V1/run_residual.py'),
                          ('run_mechanism.py',ROOT.parent/'CropMechanism-V1/run_mechanism.py'),
                          ('run_cross_spectrum.py',ROOT/'run_cross_spectrum.py')]:
    embedded+=f"(script_dir / {filename!r}).write_text({location.read_text()!r}, encoding='utf-8')\n"
cell('code',embedded)
cell('markdown', '''## 冻结的先后顺序与判据

1. 检查全部源图像哈希、原代码/模型/密钥；重放 16 个 Oracle 的原 LF/HF/联合分数。工程错误单独记录。
2. 只读取 16 个零位移样本，计算 B/C 的全部 17 钥分数。
3. **零位移门槛**：C 正样本注册钥第一名 ≥7/8；配对正样本注册钥分数高于负样本 ≥7/8；
   C 标准化间隔中位数 >0 且 ≥B 的 80%；负样本注册钥第一名 <3/8。
   B 中位数非正时记为参照不足并停止，不把无法比较当作 C 通过。
4. 只有上述门槛通过，才检查纯潜变量相位平移下 C 全部钥分数变化 ≤1e-6。
5. 通过数学检查后，仅执行 x/y 各 ±2、x/y 各 +8 RGB 像素。整数数组复制，不插值。
   黑填充和循环移位若不相同，停止而非挑选有利边界。
6. **位移门槛**：四个 ±2 条件分别要求注册钥第一名 ≥6/8、正分数高于配对负样本 ≥6/8、
   C 间隔中位数保留零位移的 ≥50%。最坏间隔变化的中位数比 B 降低 ≥30%，至少 6/8 图像改善。
   所有条件的负样本第一名均须 <3/8。8 像素仅作整格点参照，不代替 ±2 像素成功。

间隔 = 注册钥分数 − 固定 16 错钥最大分数。B/C 各自的报告尺度固定为：8 个零位移负样本中，
16 错钥分数减去本图错钥均值后的平方偏差，汇总取均值再开方（总体 RMS，不做无偏修正）。
该尺度仅标准化报告，不改变读法或密钥排名，也不是检测阈值。零/非有限尺度为工程不可用。
平局不计为第一名，正负同分不计为配对胜出；原始分数全部保留。

这些是开发筛选门槛，不是置信度或 FPR 保证。图像及其重复位移不是独立扩充样本。
''')
cell('code', '''run_logged([sys.executable, str(script), '--source', str(SOURCE), '--output', str(output),
            '--repo', str(checkout)], 'cross-spectrum-diagnostic', env=child_env, cwd=checkout)
summary = json.loads((output / 'summary.json').read_text())
print('STATUS:', summary['status'])
print('ZERO GATE:', json.dumps(summary['zero_gate'], ensure_ascii=False, indent=2))
if summary['status'].startswith('STOPPED_'):
    print('已按预定门槛停止：没有运行纯相位或实际位移对照，不自动调参重试。')
elif summary['shift_gate'] is not None:
    print('SHIFT GATE:', json.dumps(summary['shift_gate'], ensure_ascii=False, indent=2))
print('Saved result directory:', output)
''')
cell('markdown', '''## 结果检查

零位移图始终显示 B/C 的逐图标准化密钥间隔。只有完整位移对照结束，才显示位移曲线。
图中的零线仅表示注册钥与最强错钥打平，**不是经过校准的水印存在阈值**。
位移图中每条细线是一张图像，同一图像重复观测不增加独立样本数。
''')
cell('code', '''import matplotlib.pyplot as plt
rows=summary['records']
gate=summary['zero_gate']
if gate is not None:
    scales=gate['scales']
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,layout='constrained')
    for ax,role in zip(axes,('positive','negative')):
        zero=sorted([r for r in rows if r['role']==role and r['dx_rgb']==0 and r['dy_rgb']==0],key=lambda r:r['unit_id'])
        for method,marker,color,offset in [('B','o','#0072B2',-.08),('C','^','#D55E00',.08)]:
            ax.scatter([i+1+offset for i in range(8)],
                       [r['readers'][method]['gap']/scales[method] for r in zero],
                       marker=marker,color=color,label=method)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(title=f'Zero shift: {role} (8 sources)',xlabel='Source index',xticks=range(1,9))
        ax.grid(axis='y',alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Key gap / frozen wrong-key RMS')
    fig.suptitle(summary['status'])
    fig.savefig(output/'zero_key_gaps.png',dpi=150)
    plt.show()
    if summary['status'].startswith('COMPLETE_'):
        shifts=[(0,0),(-2,0),(2,0),(0,-2),(0,2),(8,0),(0,8)]
        labels=['0','x-2','x+2','y-2','y+2','x+8','y+8']
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
        for i,role in enumerate(('positive','negative')):
            for j,method in enumerate(('B','C')):
                ax=axes[i,j]
                for uid in sorted({r['unit_id'] for r in rows}):
                    data={(r['dx_rgb'],r['dy_rgb']):r for r in rows if r['unit_id']==uid and r['role']==role}
                    ax.plot(range(7),[data[s]['readers'][method]['gap']/scales[method] for s in shifts],marker='.',alpha=.6)
                ax.axhline(0,color='gray',linewidth=.8)
                ax.set_title(f'{role}, reader {method}, 8 sources')
                ax.set_xticks(range(7),labels)
                ax.grid(axis='y',alpha=.2)
                if j==0: ax.set_ylabel('Standardized key gap')
                if i==1: ax.set_xlabel('Fixed RGB shift (pixels)')
        fig.savefig(output/'shift_key_gaps.png',dpi=150)
        plt.show()
''')
cell('markdown', '''## 解释与恢复

- `STOPPED_ZERO_GATE`：零位移已不能满足保留密钥证据的要求，本候选停止，96 个实际位移与 96 个纯相位观测均标为未运行。
- `STOPPED_UNINFORMATIVE_REFERENCE`：参照基线不足以支持预定比例比较，不等于候选通过。
- `COMPLETE_CANDIDATE_REJECTED`：零位移通过但完整固定对照未满足全部门槛；不增加搜索补救。
- `COMPLETE_CANDIDATE_PASSED`：仅支持已有 8 对图像上的开发假设，不是独立验证、低 FPR 检测或完整盲几何恢复。
- `OPERATIONAL_ERROR`：工程失败，不作为科学否定；查看 `last_failure.json` 和 `execution_logs/`。

中断后沿用 SOURCE/RUN_ID 运行全部：已完成记录跳过，未终结任务的基线会重放；终结的科学结果直接返回，
不会重新打开候选。协议/代码/源文件/环境不一致会停止。不要同时写同一输出目录。
输出包含 `summary.json`、`readout.csv`、`zero_gate.json`、`replay_gate.json`、
通过后才有的 `phase_gate.json`/`shift_gate.json`、`records/`、`phase_records/`、`attempts/` 和日志。
数据缺失不会触发重新生成图像。
''')
nb={**template,'cells':cells}
nb['metadata']['colab']['name']='CEG_WM_crop_cross_spectrum_v1.ipynb'
(ROOT/'CEG_WM_crop_cross_spectrum_v1.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
