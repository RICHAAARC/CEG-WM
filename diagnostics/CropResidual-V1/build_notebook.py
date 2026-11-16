"""Self-contained residual-only Colab notebook."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
cells = []


def cell(kind, text):
    value = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """# CEG-WM：保存图像的残余平移／尺度敏感性诊断

**不生成新图像，不加载去噪器、文本编码器、DINO 或 SyncSeal 模型。仅加载原版本 VAE 评分。**

沿用刚才成功的 **Python 3.12 GPU 环境**（L4 即可），Secrets 沿用 `HF_TOKEN`、`CEG_WM_ROOT_KEY`。
运行全部单元。默认只读输入为 `MyDrive/CEG-WM/crop-bottleneck-v1/attempt01/`，
新结果写到 `MyDrive/CEG-WM/crop-residual-v1/attempt01/`。输入缺失会停止，绝不回退到生成。

固定 8 对图像，每张 22 个评分条件，共 352 条。先通过全部 32 个旧 Oracle／预测 H 重放检查，才测扰动。
CPU 工程检查不代表真实 GPU 诊断已经通过；本 Notebook 将给出新结果。
""")
cell("code", """from google.colab import drive, userdata
drive.mount('/content/drive')
import os, pathlib, subprocess, sys, json, re, time
SOURCE = pathlib.Path('/content/drive/MyDrive/CEG-WM/crop-bottleneck-v1/attempt01')
RUN_ID = 'attempt01'
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_ID)
output = pathlib.Path('/content/drive/MyDrive/CEG-WM/crop-residual-v1') / RUN_ID
checkout = pathlib.Path('/content/cegwm-crop-residual-source')
script = pathlib.Path('/content/run_crop_residual.py')
expected = 'd9ef4adb4f8424f1102f577bb76347fbba112bd9'
saved_env = json.loads((SOURCE / 'environment.json').read_text())
saved_protocol = json.loads((SOURCE / 'protocol.json').read_text())
saved_summary = json.loads((SOURCE / 'summary.json').read_text())
if saved_summary['status'] != 'COMPLETE_DIAGNOSTIC':
    raise RuntimeError('Original crop run is incomplete. No generation will be started.')
source_python = tuple(map(int, saved_env['python'].split('.')[:2]))
if sys.version_info[:2] != source_python:
    raise RuntimeError(f'Use the successful historical Colab Python {source_python[0]}.{source_python[1]} runtime; current Python is {sys.version.split()[0]}. No dependencies or source results were changed.')
child_env = dict(os.environ)
for name in ('HF_TOKEN', 'CEG_WM_ROOT_KEY'):
    value = userdata.get(name)
    if not value:
        raise RuntimeError(f'Missing secret: {name}')
    child_env[name] = value
child_env['HF_HOME'] = '/content/cegwm-crop-v1-hf'
child_env['HF_HUB_CACHE'] = child_env['HF_HOME'] + '/hub'
child_env['PYTHONUNBUFFERED'] = '1'
child_env.pop('HF_HUB_OFFLINE', None)
child_env.pop('TRANSFORMERS_OFFLINE', None)

def run_logged(command, label, **kwargs):
    # Explicit PIPE+print: Colab may hide output inherited by subprocess.run.
    log_dir = output / 'execution_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f'{label}-{time.time_ns()}.log'
    with log.open('x', encoding='utf-8') as sink:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, **kwargs) as process:
            for line in process.stdout:
                for key in ('HF_TOKEN', 'CEG_WM_ROOT_KEY'):
                    line = line.replace(child_env[key], '[REDACTED]')
                print(line, end='', flush=True)
                sink.write(line)
                sink.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f'{label} failed with exit code {code}. Full output: {log}')

run_logged([sys.executable, '-c',
    'import torch; assert torch.cuda.is_available(), "Select a GPU runtime"; '
    'print({"python":__import__("sys").version,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__})'], 'gpu-check')
print('Read-only source:', SOURCE)
print('New output:', output)
""")
cell("code", """# Match the successful source reader dependencies; preserve installed CUDA torch.
packages = saved_env['packages']
names = ('diffusers','transformers','accelerate','numpy','Pillow','huggingface_hub','safetensors','sentencepiece')
dependencies = [f'{name}=={packages[name]}' for name in names]
torch_versions = subprocess.run([sys.executable, '-c',
    'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ("torch","torchvision")}))'],
    capture_output=True, text=True, check=True)
constraints = pathlib.Path('/content/cegwm-residual-cuda-constraints.txt')
constraints.write_text(''.join(f'{name}=={ver}\\n' for name,ver in json.loads(torch_versions.stdout).items()))
run_logged([sys.executable, '-m', 'pip', 'install', '--only-binary=sentencepiece',
            '-c', str(constraints), *dependencies], 'pip-install')
if not checkout.exists():
    run_logged(['git', 'clone', 'https://github.com/RICHAAARC/CEG-WM.git', str(checkout)], 'clone')
run_logged(['git', '-C', str(checkout), 'checkout', '--detach', expected], 'checkout')
child_env['PYTHONPATH'] = str(checkout / 'src') + os.pathsep + str(checkout)
run_logged([sys.executable, '-c',
    'from diffusers import AutoencoderKL; from diffusers.image_processor import VaeImageProcessor; '
    'from cegwm.runtime.blind_scoring_v2 import _lf,_hf; '
    'print("VAE and frozen scoring imports OK; no models loaded")'], 'imports', env=child_env)
""")
cell("code", "# Self-contained residual-only script.\nscript.write_text(" + repr((ROOT / "run_residual.py").read_text(encoding="utf-8")) + ", encoding='utf-8')\n")
cell("code", """run_logged([sys.executable, str(script), '--source', str(SOURCE), '--output', str(output),
            '--repo', str(checkout)], 'residual-diagnostic', env=child_env, cwd=checkout)
summary = json.loads((output / 'summary.json').read_text())
print(json.dumps({'status':summary['status'], 'planned_scores':summary['planned_scores'],
                  'counts':summary['counts']}, indent=2))
print('Result directory:', output)
""")
cell("markdown", """## 固定对照与解释

- 4 个对照：重放原 Oracle、重放原预测 H、Oracle 不做半像素转换、实际 H 误差固定缩小一半。
- 12 个平移点：x/y 各 ±0.5、±2、±4 像素。
- 6 个各向同性尺度点：以画布中心缩放 ±0.5%、±2%、±4%。
- 所有 H 直接作用于保存的裁剪图，**只采样一次**；不对 Oracle 输出再次变换。原预测 H 不重新估计。
- 半误差点只用于判断修正实际复合误差的作用，使用真值，不能作为盲方法的性能。
- 每张负样本走相同条件。所有点分别报告，不取最大分数，不重新调阈值，不把相关重复观测当独立负样本。
- 保存全部 LF/HF 注册／16 错钥证据、相对 Oracle 的变化、有效角点误差、RGB RMSE 和 VAE 表示差异。
  内区指标固定为 RGB `[64:448]²`、latent `[8:56]²`，不按结果选择掩膜；掩膜不参与评分。

若微小扰动就明显降低正确密钥证据，说明现有读出链对错位敏感；若较大误差仍能读取，
而预测 H 不行，应进一步检查预测复合误差和坐标机制。两种原因可以同时存在。
本轮不能单独分离 VAE 非等变性与 HF 载体相位敏感性，也不能据此宣称扩散独有机制或新方法有效。

## 恢复与输出

中断后保持 SOURCE/RUN_ID 不变运行全部单元：已完成的扰动评分跳过，重启会重读 Oracle 参照以验证读出环境。
协议、源文件哈希、脚本和环境不一致会停止，不能混用旧结果。不要同时写同一输出目录。

结果：`summary.json`、`readout.csv`、`replay_gate.json`、`protocol.json`、`environment.json`、
`records/`、`attempts/`、`execution_logs/`。输入缺失、重放图像或分数不匹配会停止；不会重新生成任何源图像。
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"name": "CEG_WM_crop_residual_v1.ipynb"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "CEG_WM_crop_residual_v1.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
