"""Build self-contained mechanism notebook from the validated Colab setup."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'CropResidual-V1'
template=json.loads((PREVIOUS/'CEG_WM_crop_residual_v1.ipynb').read_text())
cells=[]
def cell(kind,text):
    item={'cell_type':kind,'id':f'cell-{len(cells):02d}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': item.update(execution_count=None,outputs=[])
    cells.append(item)

cell('markdown', '''# CEG-WM：相位敏感性与 VAE 重编码影响的最小诊断

**复用保存的 Oracle RGB，不重新生成图像。只加载原版 VAE，不加载去噪器、文本编码器或几何预测器。**

沿用成功的 **Python 3.12 GPU 环境**和 Secrets：`HF_TOKEN`、`CEG_WM_ROOT_KEY`，运行全部单元。
只读输入：`MyDrive/CEG-WM/crop-bottleneck-v1/attempt01/`；新输出：`MyDrive/CEG-WM/crop-mechanism-v1/attempt01/`。

固定 8 对图像 × 7 个整数平移条件 = **112 条记录**。先重放 16 个原 Oracle 分数，任何一张不匹配即停止。
本地只验证过工程与数学对照，尚未运行真实 GPU 诊断；本 notebook 不预填结论。

## 对照设计

RGB 平移：零位移、x/y 各 ±2 像素，以及 x/y 各 +8 像素。数组直接复制，不做 RGB 插值。
8 RGB 像素对应一个潜变量格点；2 RGB 像素对应四分之一格点。

设 `z=E(x)`、`b=E(全黑画布)`、`r=Q(z-b)`。Q 去除 64×64 偶数网格的 Nyquist 行/列，
避免分数格点 Fourier 平移后丢弃虚部而误称“纯相位”。所有反事实 HF 比较使用同一去 Nyquist 掩膜。

1. `b + FourierShift(r)`：周期边界、保范数且可逆的纯相位对照。
2. 零扩展 32/64 格后的 Fourier 平移：检查边界模型和填充收敛。
3. 周期／零扩展双线性潜变量平移：检查插值模型差异。
4. `E(整数平移 RGB)`：实际 VAE 重编码路径；黑填充与循环移位分别检查，像素相同则复用编码。
5. 对实际重编码潜变量逆移已知位移：检查正确坐标补偿后是否仍有额外偏差。
6. 不减黑画布基线的周期相位对照：检查基线建模是否左右结论。

原始 LF/HF/联合分数和历史阈值只用于实际 RGB 读出。反事实 HF 只报告连续分数和正确钥减最大错钥的间隔，
**不沿用原联合阈值、不选最优位移、不宣称盲检测性能**。
''')
setup=''.join(template['cells'][1]['source'])
setup=setup.replace('crop-residual-v1','crop-mechanism-v1').replace('cegwm-crop-residual-source','cegwm-crop-mechanism-source')
setup=setup.replace("script = pathlib.Path('/content/run_crop_residual.py')", "script_dir = pathlib.Path('/content/cegwm-crop-mechanism')\nscript_dir.mkdir(exist_ok=True)\nscript = script_dir / 'run_mechanism.py'")
cell('code',setup)
dependencies=''.join(template['cells'][2]['source']).replace('cegwm-residual-cuda-constraints','cegwm-mechanism-cuda-constraints')
cell('code',dependencies)
embedded="# Embedded, version-bound reader and mechanism script.\n"
embedded+="(script_dir / 'run_residual.py').write_text("+repr((PREVIOUS/'run_residual.py').read_text())+", encoding='utf-8')\n"
embedded+="script.write_text("+repr((ROOT/'run_mechanism.py').read_text())+", encoding='utf-8')\n"
cell('code',embedded)
cell('code', '''run_logged([sys.executable, str(script), '--source', str(SOURCE), '--output', str(output),
            '--repo', str(checkout)], 'mechanism-diagnostic', env=child_env, cwd=checkout)
summary = json.loads((output / 'summary.json').read_text())
print(json.dumps({'status':summary['status'], 'planned_records':summary['planned_records'],
                  'counts':summary['counts']}, indent=2))
print('Result directory:', output)
''')
cell('markdown', '''## 读数：先看对照，再判断机制

下图每组为相同 8 张图像：点为均值，竖线为样本最小—最大值，**不是置信区间**。
正负样本分开，不把同一图像的多次平移当作独立样本。完整 17 密钥分数在 `summary.json`。
数值恒等式：总损失 = 周期相位损失 + 边界模型差异 + 实际编码相对填充相位的额外差异。
这是有符号分数分解，**不是因果贡献比例**，负值保留。
''')
cell('code', '''import numpy as np
import matplotlib.pyplot as plt
shifts = [(0,0),(-2,0),(2,0),(0,-2),(0,2),(8,0),(0,8)]
labels = ['0','x-2','x+2','y-2','y+2','x+8','y+8']
series = [('total_loss','Actual loss','#222222','o'),
          ('phase_loss','Periodic phase','#0072B2','s'),
          ('extra_vs_padded_phase','Extra vs padded phase','#D55E00','^')]
assert summary['status'] == 'COMPLETE_DIAGNOSTIC'
fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
for i,role in enumerate(('positive','negative')):
    for j,metric in enumerate(('registered','gap')):
        ax=axes[i,j]
        for n,(field,label,color,marker) in enumerate(series):
            groups=[[r['decomposition'][metric][field] for r in summary['records']
                     if r['role']==role and (r['dx_rgb'],r['dy_rgb'])==shift] for shift in shifts]
            assert all(len(g)==8 for g in groups)
            values=np.asarray(groups)
            mean=values.mean(axis=1)
            ax.errorbar(np.arange(7)+(n-1)*.16,mean,
                        yerr=np.maximum(0,np.stack([mean-values.min(axis=1),values.max(axis=1)-mean])),
                        fmt=marker,color=color,label=label,capsize=2,linestyle='none')
        ax.axhline(0,color='gray',linewidth=.7)
        ax.set_title(f'{role}: {metric}, n=8 per shift')
        ax.set_ylabel('Signed common-HF score change')
        ax.set_xticks(range(7),labels)
        ax.grid(axis='y',alpha=.2)
        if i==1: ax.set_xlabel('Exact RGB shift (pixels)')
axes[0,0].legend(fontsize=9)
fig.suptitle('Saved Oracle images: conditional mechanism contrasts (not causal shares)')
fig.savefig(output / 'mechanism_contrasts.png',dpi=160)
plt.show()
# Fixed controls: show largest absolute score change per role; preserve every row in CSV.
controls=('boundary_model_delta','periodic_interpolation_delta','padding_32_vs_64',
          'rgb_boundary_delta','black_baseline_model_delta')
for role in ('positive','negative'):
    rows=[r for r in summary['records'] if r['role']==role]
    print(role, 'control max |registered-HF delta|:',
          {name:round(max(abs(r['decomposition']['registered'][name]) for r in rows),6) for name in controls})
    print(role, 'max removed residual energy fraction:',
          round(max(r['metrics']['nyquist_removed_energy_fraction_base_residual'] for r in rows),6))
''')
cell('markdown', '''## 可支持的结论与停止条件

- 如果纯相位已造成与实际路径接近的正确钥证据下降，而额外偏差较小、边界/插值/基线控制稳定，
  支持“当前 HF 读出相位敏感性是主要解释”。不能因此证明载体信息消失；纯相位本身可逆。
- 如果实际路径比相位及双线性模型都有明显额外下降，且已知逆移仍不能消除，支持存在
  “VAE 重编码链相对平移模型的额外非等变性”。这不等于证明 VAE 不可逆地删除水印。
- 如果 2 像素失败而 8 像素的额外偏差明显减小，提示子格点/下采样相位效应；若 8 像素仍有额外偏差，
  需考虑边界、非局部运算等，不能只归因于四分之一格点插值。
- 边界、插值、Nyquist 投影或黑基线假设若足以改变上述判断，结论记为 **未区分**；不继续增加搜索来制造阳性。
- 文件/哈希/版本/密钥不匹配、16 个旧分数重放失败、零平移或 Fourier 可逆性检查失败：立即停止并保留失败记录。
- 不自动训练、不改嵌入载体、不扩充图像、不重新估计 H、不把这一轮当作论文方法验证。

这里从保存 Oracle 图像再做整数平移，控制了新增 RGB 插值；因此不是逐像素复现上一轮“从裁剪图一次重采样”的实现。
既有 Oracle 重采样是所有路径共享的基线，本轮仅解释该基线附近的读出敏感性。

中断后保持 SOURCE/RUN_ID 不变运行全部单元；已完成记录跳过，重放检查会重做。
结果：`summary.json`、`readout.csv`、`replay_gate.json`、`protocol.json`、`environment.json`、
`records/`、`attempts/`、`execution_logs/`、`mechanism_contrasts.png`。不要同时写同一输出目录。
''')
nb={**template,'cells':cells}
nb['metadata']['colab']['name']='CEG_WM_crop_mechanism_v1.ipynb'
(ROOT/'CEG_WM_crop_mechanism_v1.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
