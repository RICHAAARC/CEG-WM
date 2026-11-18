"""Generate an output-free Colab notebook pinned to an already-created code commit."""
import argparse
from pathlib import Path
import re
import textwrap
import nbformat as nbf


def build(code_commit):
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("provide an actual full code commit SHA")
    notebook = nbf.v4.new_notebook()
    def md(text):
        notebook.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):
        notebook.cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md("""
    # CEG-WM 自参照关系水印 v1 — Colab Run all

    运行菜单选择 **Run all / 全部运行**。默认自动准备模型与 640 张图的最终 latent 缓存，
    训练 B0/B1/C 各 10000 次更新，锁定 checkpoint，执行 dev A 门及满足门控后的 B/test。
    训练可中断；重新 Run all 会验证缓存和断点，恢复未完成部分。进度与结果存入
    `/content/drive/MyDrive/CEG-WM/self-reference-v1/`。
    冻结的三臂预算可能跨越多个 Colab 会话；不保证单个会话可以跑完。

    **开始前：**选择 A100 40GB/80GB 和 high-RAM。程序要求 GPU≥38 GiB、RAM≥40 GiB，
    这只是保守运行条件，未经真实 GPU 验证，不保证一定装得下，也不支持自动改成 T4/L4 小预算。
    **存储也需预留数十 GiB**：完整 SD3.5（含三个文本编码器）持久保存在 Drive，另留 8 GiB
    给 VAE 副本、缓存、断点和 LPIPS。下载前按 HF 官方文件大小计算缺失字节并检查可用空间，
    不足会先停止；普通 15GB Drive 通常不够。DriveFS 估计不能替代共享账号真实配额。
    请先在 [SD3.5 Medium 模型页面](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
    接受许可，并将有访问权限的 HF_TOKEN 放入 Colab Secrets；没有 Secret 时会隐藏输入。
    必要交互仅 Google Drive 授权和 HF 许可/token。Token 不写 Drive、不打印、不进入命令行。

    **此 Notebook 未在真实 Colab/GPU 上执行。**已做格式/编译及 CPU/fake 控制流验证，
    不包含伪造实验输出。640 张图是采样单位，80 个主题 family 防跨 split 模板泄漏并用于
    bootstrap 聚类；不把这些图称为统计独立。三臂训练为正式初始预算，可能耗时较长。
    """)
    md("### 1. 挂载 Drive，检出不可变代码\n打开链接所用分支可以更新；实际执行代码固定到下方 SHA。")
    code(f'''
    import os, sys, json, subprocess, time, getpass, re
    from pathlib import Path
    BOOT_OK = False
    controller = None
    SETUP_ERROR = None
    CODE_COMMIT = {code_commit!r}
    BRANCH = "research/self-reference-relation-v1"
    BASE = Path("/content/drive/MyDrive/CEG-WM/self-reference-v1")
    REPO = Path("/content/cegwm-self-reference-" + CODE_COMMIT[:12])
    VENV = Path("/content/cegwm-sr-venv-" + CODE_COMMIT[:12])
    PYTHON = VENV / "bin/python"
    def setup_redact(value):
        text = str(value)
        secret = os.environ.get("HF_TOKEN")
        if secret:
            text = text.replace(secret, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{{8,}}", "[REDACTED]", text)
    os.environ.update({{"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HOME": "/content/cegwm-hf-home",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}})
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE.mkdir(parents=True, exist_ok=True)
        if not (REPO / ".git").exists():
            subprocess.run(["git", "clone", "--branch", BRANCH, "https://github.com/RICHAAARC/CEG-WM.git", str(REPO)], check=True)
        # Existing immutable checkout is reused, never reset/stashed over user edits.
        current = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        if current != CODE_COMMIT:
            if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
                raise RuntimeError("Checkout contains edits; cannot switch its immutable code identity")
            subprocess.run(["git", "-C", str(REPO), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_COMMIT], check=True)
        sys.path.insert(0, str(REPO / "src"))
        from cegwm.self_reference.colab_runner import run_logged, initialize, Controller
        from cegwm.self_reference.colab_storage import atomic_json, redact
        from cegwm.self_reference.colab_config import DEPENDENCIES
        BOOT_OK = True
        print("Code:", CODE_COMMIT)
    except Exception as error:
        BOOT_OK = False
        SETUP_ERROR = setup_redact(type(error).__name__ + ": " + str(error))
        print("STOP:", SETUP_ERROR)
        if BASE.exists():
            (BASE / "setup_error.json").write_text(json.dumps({{"status": "error", "reason": SETUP_ERROR}}, indent=2))
    ''')
    md("""
    ### 2. 独立环境与访问凭据
    复用 Colab 自带 torch/torchvision；安装命令全部 `--no-deps`，不会间接替换它们。
    pip 日志可见并保存在 Drive。安装后子进程实际导入 pipeline、三个编码器接口、VAE、
    sentencepiece 和 LPIPS；不加载权重。不兼容会明确停止，不做静默降级。
    """)
    code('''
    if BOOT_OK:
        try:
            bootstrap_log = BASE / "bootstrap.log"
            if not PYTHON.exists():
                run_logged([sys.executable, "-m", "venv", "--system-site-packages", str(VENV)], bootstrap_log)
            version_probe = "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['torch','torchvision']}))"
            system_versions = json.loads(subprocess.check_output([sys.executable, "-c", version_probe], text=True))
            constraints = VENV / "system-torch-constraints.txt"
            constraints.write_text("\\n".join(k + "==" + v for k,v in system_versions.items()) + "\\n")
            run_logged([PYTHON, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "-c", constraints, *DEPENDENCIES], bootstrap_log)
            run_logged([PYTHON, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", REPO], bootstrap_log)
            actual_versions = json.loads(subprocess.check_output([str(PYTHON), "-c", version_probe], text=True))
            if actual_versions != system_versions:
                raise RuntimeError("torch/torchvision changed unexpectedly")
            smoke = "import torch,torchvision,sentencepiece,lpips; from diffusers import StableDiffusion3Pipeline,AutoencoderKL; from transformers import T5EncoderModel,CLIPTextModel,CLIPTextModelWithProjection; print('Dependency import smoke passed; no weights loaded')"
            run_logged([PYTHON, "-c", smoke], bootstrap_log)
            token = None
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                pass
            if not token:
                token = getpass.getpass("HF_TOKEN (hidden; complete cached runs may leave blank): ")
            if token and token.strip():
                os.environ["HF_TOKEN"] = token.strip()
            else:
                os.environ.pop("HF_TOKEN", None)
            del token
            RUN = initialize(REPO, BASE, PYTHON, CODE_COMMIT)
            controller = Controller(RUN)
            print("Drive run:", RUN)
        except Exception as error:
            BOOT_OK = False
            SETUP_ERROR = redact(type(error).__name__ + ": " + str(error))
            atomic_json(BASE / "setup_error.json", {"status": "error", "reason": SETUP_ERROR})
            print("STOP:", SETUP_ERROR)
    else:
        print("SKIPPED: Drive/code setup failed")
    ''')
    md("""
    ### 3. 准备或复用缓存
    先验证每源完成标记、哈希、固定身份和 VAE。完整缓存不加载或重新下载 denoiser/T5。
    缺 latent 时一个独立进程加载模型一次、逐源写入；随后另一个进程加载 float32 VAE 生成
    同 z 的普通 RGB8。每个文件与 sidecar 都完整才视为完成。HF gate 先于大型下载检查。
    """)
    code('''
    if BOOT_OK and controller is not None:
        controller.execute("cache")
    else:
        print("SKIPPED cache: setup did not complete")
    ''')
    md("""
    ### 4. 三臂训练与断点恢复
    每臂独立进程，float32 VAE；microbatch=1，4 次 loss/4 backward 后一次 AdamW，
    effective batch=4、10000 updates 不变。decode/encode 使用显式 nonreentrant checkpoint。
    每 100 步留恢复快照，每 1000 步留固定选模候选。坏/半写快照隔离保留，退回有效断点；
    缺少历史固定候选则从该候选之前重放，不能只凭 final 快照跳过。
    """)
    code('''
    if BOOT_OK and controller is not None:
        controller.execute("train")
    else:
        print("SKIPPED train: setup did not complete")
    ''')
    md("""
    ### 5. 锁定 checkpoint，先 dev A，再条件 B/test
    只使用 model_selection 选 checkpoint。dev A 失败会持久化原因，**不执行非零位移评估或 test**；
    训练中的固定位移增强始终保留。通过后才执行四个精确 ±2px 位移和最终 test。
    缺/坏 LPIPS 或质量预算不通过不能报告研究通过。A/B 判定不是新颖性或低 FPR 证明。
    """)
    code('''
    if BOOT_OK and controller is not None:
        controller.execute("evaluate")
    else:
        print("SKIPPED evaluation: setup did not complete")
    ''')
    md("""
    ### 6. 查看状态与结果路径
    `execution_completed` 仅代表流程结束；科学状态单独显示。失败/中断或 A 停止不会显示完成/PASS。
    Drive 保存 runtime、code/model/config/manifest 哈希、训练断点、逐源结果、固定样图和实时日志。
    重连后重新 Run all 即可继续验证后的同一版本；不要手工改 split、预算或 checkpoint。
    """)
    code('''
    if controller is not None:
        summary = controller.summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("NOT COMPLETED:", SETUP_ERROR or "setup has not run")
    ''')
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}, "accelerator": "GPU",
                         "colab": {"name": "CEG_WM_SelfReference_V1_Colab.ipynb", "provenance": []},
                         "cegwm": {"code_commit": code_commit, "validation": "CPU/fake and static only; no real Colab execution"}}
    nbf.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{index}", "exec")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", default="notebooks/CEG_WM_SelfReference_V1_Colab.ipynb")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(args.code_commit), path)
    print(path)
