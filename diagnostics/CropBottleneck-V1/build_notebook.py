"""Build a self-contained upload-to-Colab notebook; no publication or network."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True),
                  "id": f"cell-{len(cells):02d}"})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {},
                  "source": text.splitlines(True), "id": f"cell-{len(cells):02d}"})


md("""# CEG-WM 裁剪瓶颈：8 对样本最小诊断

仅运行固定前 8 个正式源的重生成与四路径评分，**不运行正式 worker、不训练、不重新校准、不搜索变换**。
这是新一轮机制诊断，不保证重现历史 PNG 字节或历史分数。

运行前：选择 GPU 运行时（建议 A100 和高 RAM；原管线完整驻留 GPU），在 Colab Secrets 中提供
`HF_TOKEN`、`CEG_WM_ROOT_KEY` 并允许本 notebook 访问。HF 账户需已有 SD3.5 模型访问权限。
然后运行全部单元。首次会下载模型。代码只使用固定历史提交，诊断脚本已内嵌，无须另行上传 Python 文件。

结果保存在 Drive `CEG-WM/crop-bottleneck-v1/attempt01/`。中断后使用相同 RUN_ID 再次运行全部单元；
已提交的成对 PNG 和已完成评分会校验哈希并跳过，失败和中断尝试保留。
换依赖、GPU 型号、模型版本或诊断脚本时会拒绝混合续跑；要另设 RUN_ID。

**本入口已做本地工程检查，尚未执行真实 Colab/GPU 实验。不要据此宣称水印或 Oracle 已通过。**
""")
code("""from google.colab import drive, userdata
drive.mount('/content/drive')
import os, pathlib, subprocess, sys, json, re
RUN_ID = 'attempt01'
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_ID)
output = pathlib.Path('/content/drive/MyDrive/CEG-WM/crop-bottleneck-v1') / RUN_ID
checkout = pathlib.Path('/content/cegwm-crop-v1-source')
runtime = pathlib.Path('/content/cegwm-crop-v1-runtime')
script = pathlib.Path('/content/run_crop_diagnostic.py')
expected = 'd9ef4adb4f8424f1102f577bb76347fbba112bd9'
child_env = dict(os.environ)
for name in ('HF_TOKEN', 'CEG_WM_ROOT_KEY'):
    value = userdata.get(name)
    if not value:
        raise RuntimeError(f'Missing Colab secret: {name}')
    child_env[name] = value
child_env['HF_HOME'] = '/content/cegwm-crop-v1-hf'
child_env['HF_HUB_CACHE'] = child_env['HF_HOME'] + '/hub'
child_env['TOKENIZERS_PARALLELISM'] = 'false'
child_env['PYTHONUNBUFFERED'] = '1'
# Check in a child: pip changes below cannot leave stale imported modules in the kernel.
subprocess.run([sys.executable, '-c',
    'import torch; assert torch.cuda.is_available(), "Select a Colab GPU runtime"; '
    'print({"gpu":torch.cuda.get_device_name(0),"vram_GiB":round(torch.cuda.get_device_properties(0).total_memory/2**30,1)})'], check=True)
""")
code("""# Preserve Colab's matched CUDA torch/torchvision. These are preparation pins,
# not a claim that this exact full-model environment has already passed.
dependencies = ['diffusers==0.39.0', 'transformers==4.57.6', 'accelerate==1.1.1',
                'sentencepiece==0.2.0', 'protobuf>=3.20']
subprocess.run([sys.executable, '-m', 'pip', 'install', *dependencies], check=True)
if not checkout.exists():
    subprocess.run(['git', 'clone', 'https://github.com/RICHAAARC/CEG-WM.git', str(checkout)], check=True)
actual = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                        capture_output=True, text=True, check=True).stdout.strip()
if actual != expected:
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', expected], check=True)
child_env['PYTHONPATH'] = str(checkout / 'src') + os.pathsep + str(checkout)
subprocess.run([sys.executable, '-c',
    'from cegwm.shared.keys import normalize_detection_key,public_key_digest; '
    'from cegwm.protocol.content_chain import CONTENT_CHAIN_PUBLIC_KEY_DIGEST; '
    'import os; assert public_key_digest(normalize_detection_key(os.environ["CEG_WM_ROOT_KEY"])) == CONTENT_CHAIN_PUBLIC_KEY_DIGEST, "Detection key identity differs"; '
    'from diffusers import StableDiffusion3Pipeline; '
    'from transformers import AutoModel, AutoImageProcessor; '
    'print("Imports and key identity OK; no models loaded")'], env=child_env, check=True)
""")
code("# Self-contained diagnostic implementation, identical to delivered Python file.\n"
     + "script.write_text(" + repr((ROOT / "run_crop_diagnostic.py").read_text(encoding="utf-8"))
     + ", encoding='utf-8')\n")
code("""command = [sys.executable, str(script), '--repo', str(checkout), '--output', str(output), '--runtime', str(runtime)]
prepare_env = {k:v for k,v in child_env.items() if k not in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
subprocess.run(command + ['--prepare-assets'], env=prepare_env, check=True)
# Model SHA is frozen in Drive; historical loaders use only the isolated pinned cache.
run_env = dict(child_env, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
completed = subprocess.run(command, env=run_env, cwd=checkout, check=False)
summary_path = output / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(json.dumps({'status': summary['status'], 'counts': summary['counts']}, indent=2))
print('Output directory:', output)
if completed.returncode:
    raise RuntimeError('Diagnostic stopped. Completed PNGs/scores are retained; inspect the error above and rerun after fixing the cause.')
""")
md("""## 结果解释与停止边界

- A=`clean`，B=`crop_direct`，C=`predicted_h`（仅校正后分数），D=`oracle_h`。
- C 不执行原流程的直读阳性提前返回，以便每张图都获得校正后证据；原流程的判定可由 B/C 重建。
- 负样本是同种子 primary null，无 SyncSeal；与旧正式配对一致。
- `summary.json` 固定保留 8 个源、64 个路径槽位。缺失、无合法 H、运行失败分别记录，不改分母。
- `above_historical_reference_tau` 仅是旧阈值参照，不是重新校准后的 TPR/FPR。
- Oracle 仅使用攻击真值，不读取原图补缺失像素。几何误差用物理像素中心矩阵；
  Oracle 传给原 PIL 渲染器的矩阵另做固定半像素坐标转换，二者均落盘。
- 首先检查 A 的可读性；D 好而 C 差支持同步问题；C/D 都差才进一步检查观测或载体。
  **D 失败不证明信息论意义上的载体消失。** 不自动启动掩膜搜索、传播探针或新方法开发。
- `model_revisions.json` 冻结此次重跑的公开模型 SHA；历史模型版本未能确认，因此本轮不是逐字节历史复现。

完成后提供整个结果目录，至少保留 `protocol.json`、`environment.json`、`model_revisions.json`、
`syncseal_identity.json`、`summary.json` 和 `units/`；不要只保留汇总表。
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "language_info": {"name": "python"}, "colab": {"name": "CEG_WM_crop_bottleneck_v1.ipynb"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}
(ROOT / "CEG_WM_crop_bottleneck_v1.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
