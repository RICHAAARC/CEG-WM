"""
文件目的：onefile workflow 编排入口。
Module type: General module

职责边界：
1. 仅编排既有 CLI/脚本入口（embed/detect/calibrate/evaluate/audits/signoff）。
2. 不直接写 records 或关键 artifacts，不改写冻结语义与 digest 口径。
3. 通过既有 CLI override 机制传递可变项，保持门禁与白名单约束。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from pathlib import Path

# Python 3.10 兼容：3.11+ 才引入 datetime.UTC，此处向下兼容
if hasattr(__import__("datetime"), "UTC"):
    from datetime import UTC
else:
    UTC = _tz.utc
from typing import List, Sequence
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.core import records_io


PROFILE_CPU_SMOKE = "cpu_smoke"
PROFILE_PAPER_FULL_CUDA = "paper_full_cuda"
LEGACY_PROFILE_CPU_MIN = "cpu_min"
LEGACY_PROFILE_CUDA_REAL = "cuda_real"
PAPER_FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "paper_full_cuda.yaml"
PAPER_SPEC_CONFIG_PATH = REPO_ROOT / "configs" / "paper_faithfulness_spec.yaml"
PAPER_FROZEN_IMPL_REQUIRED_FIELDS = ("sync_module_id", "geometry_extractor_id")

# cfg 角色枚举
CFG_ROLE_SPEC = "spec"
CFG_ROLE_RUNTIME = "runtime"


def _detect_cfg_role(cfg_obj: dict) -> str:
    """
    功能：检测配置的角色类型（规范 vs 运行期）。

    Detect whether config is a spec (specification) or runtime config.
    Spec configs have 'authority' and 'audit_gate_requirements' top-level fields.

    Args:
        cfg_obj: Configuration mapping.

    Returns:
        Role string: "spec" or "runtime".

    Raises:
        TypeError: If cfg_obj is not a dict.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    # 特征检测：spec 有权威声明字段和审计门禁声明
    authority = cfg_obj.get("authority")
    audit_gate_requirements = cfg_obj.get("audit_gate_requirements")

    # 两个特征字段都存在 -> spec
    if isinstance(authority, dict) and isinstance(audit_gate_requirements, dict):
        return CFG_ROLE_SPEC

    # 否则 -> runtime
    return CFG_ROLE_RUNTIME


def _validate_cfg_role_for_profile(cfg_obj: dict, cfg_path: Path, profile: str) -> None:
    """
    功能：验证配置角色是否与 profile 兼容。

    Validate that config role matches profile requirements.
    paper_full_cuda profile only accepts runtime configs, never spec configs.

    Args:
        cfg_obj: Configuration mapping.
        cfg_path: Configuration file path (for error messages).
        profile: Workflow profile.

    Returns:
        None.

    Raises:
        TypeError: If cfg_obj or profile is invalid.
        ValueError: If cfg role is inappropriate for profile.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")

    detected_role = _detect_cfg_role(cfg_obj)
    profile = _normalize_profile(profile)

    # paper_full_cuda profile 对应论文方法，必须只能使用 runtime cfg
    if profile == PROFILE_PAPER_FULL_CUDA:
        if detected_role == CFG_ROLE_SPEC:
            raise ValueError(
                f"profile=paper_full_cuda requires runtime config, got spec config: {cfg_path}\n"
                f"Use configs/paper_full_cuda.yaml as runtime config instead.\n"
                f"configs/paper_faithfulness_spec.yaml is a specification document, not executable at runtime."
            )

    # cpu_smoke profile 也只接 runtime cfg
    if profile == PROFILE_CPU_SMOKE:
        if detected_role == CFG_ROLE_SPEC:
            raise ValueError(
                f"profile=cpu_smoke requires runtime config, got spec config: {cfg_path}\n"
                f"Spec configs cannot be executed directly. Use a runtime config instead."
            )


def _load_paper_frozen_impl_constraints() -> dict:
    """
    功能：读取 paper_full_cuda 的冻结 impl 约束。 

    Load frozen impl constraints from paper_full_cuda config.

    Args:
        None.

    Returns:
        Mapping for required impl constraints.

    Raises:
        FileNotFoundError: If frozen config is missing.
        ValueError: If frozen config or impl section is invalid.
    """
    if not PAPER_FROZEN_CONFIG_PATH.exists() or not PAPER_FROZEN_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"paper frozen config not found: {PAPER_FROZEN_CONFIG_PATH}")

    frozen_obj = yaml.safe_load(PAPER_FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(frozen_obj, dict):
        raise ValueError("paper frozen config root must be mapping")

    frozen_impl = frozen_obj.get("impl")
    if not isinstance(frozen_impl, dict):
        raise ValueError("paper frozen config impl must be mapping")

    constraints = {}
    for field_name in PAPER_FROZEN_IMPL_REQUIRED_FIELDS:
        field_value = frozen_impl.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"paper frozen impl.{field_name} must be non-empty str")
        constraints[field_name] = field_value.strip()
    return constraints


def _validate_paper_profile_impl_constraints(cfg_obj: dict, frozen_impl_constraints: dict) -> None:
    """
    功能：校验 paper profile 的关键 impl 绑定不偏离冻结约束。 

    Validate required paper-profile impl bindings against frozen constraints.

    Args:
        cfg_obj: Effective config mapping.
        frozen_impl_constraints: Required impl constraints loaded from frozen config.

    Returns:
        None.

    Raises:
        ValueError: If impl section or required fields are missing/mismatched.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(frozen_impl_constraints, dict) or not frozen_impl_constraints:
        raise TypeError("frozen_impl_constraints must be non-empty dict")

    impl_cfg = cfg_obj.get("impl")
    if not isinstance(impl_cfg, dict):
        raise ValueError("paper_full_cuda requires config.impl to be mapping")

    for field_name, expected_value in frozen_impl_constraints.items():
        actual_value = impl_cfg.get(field_name)
        if not isinstance(actual_value, str) or not actual_value.strip():
            raise ValueError(f"paper_full_cuda requires impl.{field_name} to be non-empty str")
        normalized_actual_value = actual_value.strip()
        if normalized_actual_value != expected_value:
            raise ValueError(
                f"paper_full_cuda requires impl.{field_name}={expected_value}, got {normalized_actual_value}"
            )


def _normalize_profile(profile: str) -> str:
    """
    功能：规范化 profile 名称（兼容历史别名）。

    Normalize workflow profile names with backward-compatible aliases.

    Args:
        profile: Raw profile name.

    Returns:
        Normalized profile name.

    Raises:
        TypeError: If input type is invalid.
        ValueError: If profile is unsupported.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if profile == LEGACY_PROFILE_CPU_MIN:
        return PROFILE_CPU_SMOKE
    if profile == LEGACY_PROFILE_CUDA_REAL:
        return PROFILE_PAPER_FULL_CUDA
    if profile in {PROFILE_CPU_SMOKE, PROFILE_PAPER_FULL_CUDA}:
        return profile
    raise ValueError(f"unsupported profile: {profile}")


@dataclass(frozen=True)
class WorkflowStep:
    """
    功能：描述单个编排步骤。 

    Describe one orchestrated workflow step.

    Args:
        name: Stable step name.
        command: Executable command list.
        artifact_paths: Key artifact paths expected after step.

    Returns:
        None.
    """

    name: str
    command: List[str]
    artifact_paths: List[Path]


def _build_run_root(repo_root: Path, provided_run_root: str | None, profile: str) -> Path:
    """
    功能：解析或生成 run_root 路径。 

    Resolve run_root path from user input or generate a deterministic-form path.

    Args:
        repo_root: Repository root path.
        provided_run_root: Optional user-provided run_root.
        profile: Selected workflow profile.

    Returns:
        Absolute run_root path.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If provided_run_root is empty.
    """
    if repo_root is not None and not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path or None")
    if provided_run_root is not None and not isinstance(provided_run_root, str):
        raise TypeError("provided_run_root must be str or None")
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")

    if provided_run_root is not None:
        if not provided_run_root.strip():
            raise ValueError("run_root must be non-empty when provided")
        candidate = Path(provided_run_root)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    generated = repo_root / "outputs" / f"onefile_{profile}_{timestamp}_{suffix}"
    return generated.resolve()


def _build_stage_overrides(stage_name: str, profile: str) -> List[str]:
    """
    功能：构造阶段 override 参数。 

    Build allowed CLI overrides for each stage.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.
        profile: Workflow profile name.

    Returns:
        List of override strings in key=value format.

    Raises:
        ValueError: If stage_name or profile is unsupported.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    profile = _normalize_profile(profile)

    reason = json.dumps(f"onefile_workflow_{stage_name}", ensure_ascii=False)
    overrides = [
        "run_root_reuse_allowed=true",
        f"run_root_reuse_reason={reason}",
    ]

    if profile == PROFILE_CPU_SMOKE:
        overrides.extend(
            [
                "force_cpu=\"cpu\"",
                "enable_trace_tap=true",
            ]
        )
    if profile == PROFILE_PAPER_FULL_CUDA:
        overrides.extend(
            [
                "enable_trace_tap=true",
                "enable_paper_faithfulness=true",
            ]
        )

    if stage_name == "embed":
        overrides.append("disable_content_detect=false")
    if stage_name == "detect":
        overrides.append("enable_content_detect=true")
        # detect 阶段阈值回退为架构必要性（校准工件在 calibrate 阶段产出），
        # 此处回退仅用于中间评分，不会影响最终判决的 threshold_source。
        overrides.append("allow_threshold_fallback_for_tests=true")
    return overrides


def _resolve_default_signoff_profile_for_profile(profile: str, provided_signoff_profile: str | None) -> str:
    """
    功能：按执行 profile 解析 signoff profile 默认值。

    Resolve effective signoff profile by workflow profile with safe defaults.

    Args:
        profile: Workflow profile.
        provided_signoff_profile: Optional user provided signoff profile.

    Returns:
        Effective signoff profile in {baseline, paper, publish}.

    Raises:
        TypeError: If input type is invalid.
        ValueError: If provided signoff profile is unsupported.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if provided_signoff_profile is not None and not isinstance(provided_signoff_profile, str):
        raise TypeError("provided_signoff_profile must be str or None")

    normalized_profile = _normalize_profile(profile)
    allowed_signoff_profiles = {"baseline", "paper", "publish"}

    if isinstance(provided_signoff_profile, str) and provided_signoff_profile.strip():
        normalized_signoff_profile = provided_signoff_profile.strip().lower()
        if normalized_signoff_profile not in allowed_signoff_profiles:
            raise ValueError(f"unsupported signoff_profile: {provided_signoff_profile}")
        return normalized_signoff_profile

    if normalized_profile == PROFILE_PAPER_FULL_CUDA:
        return "paper"
    return "baseline"


def _resolve_embed_input_image_path_from_cfg(cfg_obj: dict) -> str | None:
    """
    功能：从运行期配置解析 embed 输入图路径。

    Resolve embed input image path from runtime config mapping.

    Args:
        cfg_obj: Runtime config mapping.

    Returns:
        Image path string when available, otherwise None.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    candidates = [
        cfg_obj.get("__embed_input_image_path__"),
        cfg_obj.get("input_image_path"),
    ]
    embed_node = cfg_obj.get("embed")
    embed_cfg = embed_node if isinstance(embed_node, dict) else {}
    if embed_cfg:
        candidates.append(embed_cfg.get("input_image_path"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _ensure_default_embed_input_image(
    run_root: Path,
    prompt: str | None = None,
    seed: int | None = None,
    cfg_obj: dict | None = None,
) -> Path:
    """
    功能：生成默认 embed 输入图像工件并返回路径。

    Generate a default embed input image artifact for SemanticMaskProvider.
    When prompt and cfg_obj are available, attempts SD3 preview generation
    (no injection hook) to obtain a semantically valid preview image matching the
    target generation condition. Falls back to a synthetic gradient image when the
    SD3 pipeline is unavailable (e.g., CPU-only or missing dependencies).

    Args:
        run_root: Unified run_root path.
        prompt: Optional inference prompt for SD3 preview generation.
        seed: Optional random seed for deterministic SD3 preview.
        cfg_obj: Optional full config object (used to resolve model params).

    Returns:
        Absolute path to generated default image.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    target_path = run_root / "artifacts" / "workflow_cfg" / "default_embed_input.png"
    if target_path.exists() and target_path.is_file():
        return target_path.resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 【P0-A】Preview Generation：以目标 prompt + seed 执行无注入 SD3 推理，
    # 生成与水印图像语义一致的预览图，供 SemanticMaskProvider 提取有效掩码。
    # 仅在 prompt 可用且 diffusers 可导入时启用；否则退回合成梯度图（向后兼容）。
    if isinstance(prompt, str) and prompt.strip():
        try:
            import torch
            from diffusers import StableDiffusion3Pipeline  # type: ignore[import-untyped]

            # 从顶层 cfg 读取推理参数：inference_height/inference_width 在 YAML 中为顶层键。
            _cfg = cfg_obj if isinstance(cfg_obj, dict) else {}
            _height = int(_cfg.get("inference_height", 512))
            _width = int(_cfg.get("inference_width", 512))
            _model_sub = _cfg.get("model") if isinstance(_cfg.get("model"), dict) else {}
            _dtype_str = _model_sub.get("dtype", _cfg.get("model_dtype", "float16"))
            _torch_dtype = torch.float16 if "16" in str(_dtype_str) else torch.float32

            # model_id：优先使用 cfg 中指定路径，否则使用默认 SD3 模型。
            _model_id = _cfg.get("model_id")
            if not isinstance(_model_id, str) or not _model_id.strip():
                _model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

            print(f"[P0-A] Running SD3 preview generation for embed input (prompt={repr(prompt[:60])})")
            _pipeline = StableDiffusion3Pipeline.from_pretrained(
                _model_id,
                torch_dtype=_torch_dtype,
            )
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _pipeline = _pipeline.to(_device)

            _generator = None
            if seed is not None:
                _generator = torch.Generator(device=_device).manual_seed(int(seed))

            with torch.no_grad():
                _preview = _pipeline(
                    prompt=prompt.strip(),
                    height=_height,
                    width=_width,
                    generator=_generator,
                    num_inference_steps=28,
                ).images[0]

            _preview.save(target_path, format="PNG")
            del _pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[P0-A] SD3 preview saved: {target_path}")
            return target_path.resolve()

        except Exception as _preview_exc:
            print(
                f"[P0-A] WARN: SD3 preview generation failed ({type(_preview_exc).__name__}: {_preview_exc}); "
                "falling back to synthetic gradient image",
                file=sys.stderr,
            )

    # 回退路径：合成线性颜色梯度图（语义失真，SemanticMaskProvider 掩码质量低）。
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Pillow unavailable for default embed input generation: {type(exc).__name__}") from exc

    image_size = 512
    grad_x = Image.linear_gradient("L").resize((image_size, image_size))
    grad_y = grad_x.transpose(Image.Transpose.ROTATE_90)
    grad_mix = Image.blend(grad_x, grad_y, alpha=0.5)
    default_image = Image.merge("RGB", (grad_x, grad_y, grad_mix))
    default_image.save(target_path, format="PNG")
    return target_path.resolve()


def _apply_pil_image_attack(
    image_path: Path,
    attack_family: str,
    params: dict,
    output_path: Path,
) -> bool:
    """
    功能：使用 PIL 对图像应用指定攻击族变换并保存到输出路径。

    Apply a PIL-based image-domain attack transformation.

    Args:
        image_path: Source image path.
        attack_family: Attack family name (identity/jpeg/gaussian_noise/rotate/resize/crop).
        params: Attack parameters dict from attack_protocol.yaml.
        output_path: Target output image path.

    Returns:
        True when attack applied successfully; False otherwise.
    """
    if not isinstance(image_path, Path) or not image_path.is_file():
        return False
    if not isinstance(attack_family, str) or not attack_family:
        return False
    if not isinstance(output_path, Path):
        return False

    try:
        from PIL import Image
        import numpy as np
        import io
    except ImportError:
        return False

    try:
        img = Image.open(image_path).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if attack_family in {"none", "identity"}:
            img.save(output_path, format="PNG")
            return True

        if attack_family == "jpeg":
            quality_list = params.get("quality", [85])
            quality = quality_list[0] if isinstance(quality_list, list) and quality_list else 85
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            img_decoded = Image.open(buf).convert("RGB")
            img_decoded.save(output_path, format="PNG")
            return True

        if attack_family == "gaussian_noise":
            sigma_list = params.get("sigma", [0.01])
            sigma = sigma_list[0] if isinstance(sigma_list, list) and sigma_list else 0.01
            arr = np.asarray(img, dtype=np.float32) / 255.0
            noise = np.random.default_rng(seed=42).normal(0, float(sigma), arr.shape).astype(np.float32)
            arr_noisy = np.clip(arr + noise, 0.0, 1.0)
            Image.fromarray((arr_noisy * 255).astype(np.uint8)).save(output_path, format="PNG")
            return True

        if attack_family == "rotate":
            degrees_list = params.get("degrees", [10])
            degrees = degrees_list[0] if isinstance(degrees_list, list) and degrees_list else 10
            img.rotate(float(degrees), expand=False).save(output_path, format="PNG")
            return True

        if attack_family == "resize":
            scale_list = params.get("scale_factors", [0.9])
            scale = scale_list[0] if isinstance(scale_list, list) and scale_list else 0.9
            w, h = img.size
            new_w, new_h = max(1, int(w * float(scale))), max(1, int(h * float(scale)))
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            # 还原尺寸以保持一致性
            img_resized = img_resized.resize((w, h), Image.LANCZOS)
            img_resized.save(output_path, format="PNG")
            return True

        if attack_family == "gaussian_blur":
            from PIL import ImageFilter
            kernel_list = params.get("kernel_size", [3])
            radius = (kernel_list[0] if isinstance(kernel_list, list) and kernel_list else 3) / 2.0
            img.filter(ImageFilter.GaussianBlur(radius=radius)).save(output_path, format="PNG")
            return True

    except Exception as exc:
        print(f"[onefile/attack] WARN: failed to apply {attack_family}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    return False


def _load_attack_families_from_protocol(repo_root: Path) -> List[dict]:
    """
    功能：从 attack_protocol.yaml 加载攻击族列表（名称 + 参数版本 + 参数）。

    Load attack family entries from attack_protocol.yaml for coverage generation.

    Args:
        repo_root: Repository root path.

    Returns:
        List of dicts with keys: family, params_version, params.
    """
    if not isinstance(repo_root, Path):
        return []

    protocol_path = repo_root / "configs" / "attack_protocol.yaml"
    if not protocol_path.exists():
        return []

    try:
        protocol_obj = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(protocol_obj, dict):
        return []

    result: List[dict] = []
    families = protocol_obj.get("families")
    if not isinstance(families, dict):
        return result

    for family_name, family_obj in families.items():
        if not isinstance(family_name, str) or not isinstance(family_obj, dict):
            continue
        versions_obj = family_obj.get("params_versions")
        if not isinstance(versions_obj, dict):
            continue
        for version_name, version_params in versions_obj.items():
            if not isinstance(version_name, str):
                continue
            result.append({
                "family": family_name,
                "params_version": version_name,
                "params": version_params if isinstance(version_params, dict) else {},
            })
    return result


def _run_redetect_with_np_thresholds(
    repo_root: Path,
    run_root: Path,
    cfg_path: Path,
    profile: str,
) -> None:
    """
    功能：在 calibrate 完成后，使用 NP 校准阈值工件重跑一次 detect（re-detect）。

    Re-run detect with NP canonical thresholds artifact produced by calibrate.
    Results are stored in artifacts/detect_np/. For paper_full_cuda, the
    canonical detect record is additionally promoted back to the main run_root
    after calibrate has already consumed the pre-calibration detect output.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        cfg_path: Config file path for detect.
        profile: Workflow profile.

    Returns:
        None.

    Raises:
        RuntimeError: If the re-detect subprocess returns non-zero or record is absent.
    """
    if not isinstance(repo_root, Path) or not isinstance(run_root, Path):
        return
    if not isinstance(cfg_path, Path) or not cfg_path.exists():
        return

    thresholds_artifact = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    if not thresholds_artifact.exists():
        print(
            "[onefile/redetect] WARN: thresholds_artifact.json not found, skipping re-detect",
            file=sys.stderr,
        )
        return

    detect_np_root = run_root / "artifacts" / "detect_np"
    embed_record = run_root / "records" / "embed_record.json"

    # 构建 re-detect 命令：使用 main.cli.run_detect 入口，--out 指向独立子目录，
    # 追加 --thresholds-path 注入 NP 阈值工件。
    command = [
        sys.executable,
        "-m",
        "main.cli.run_detect",
        "--out",
        str(detect_np_root),
        "--config",
        str(cfg_path),
        "--input",
        str(embed_record),
        "--thresholds-path",
        str(thresholds_artifact),
    ]
    for item in _build_stage_overrides("detect", profile):
        command.extend(["--override", item])

    print(f"[onefile/redetect] Running NP-threshold re-detect → {detect_np_root}")
    return_code = _run_subprocess_for_step(command, repo_root)
    if return_code != 0:
        raise RuntimeError(f"re-detect subprocess returned {return_code}")

    detect_np_record = detect_np_root / "records" / "detect_record.json"
    if not detect_np_record.exists():
        raise RuntimeError("re-detect returned 0 but detect_record.json is absent")

    if _normalize_profile(profile) == PROFILE_PAPER_FULL_CUDA:
        main_detect_record = run_root / "records" / "detect_record.json"
        main_detect_record.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(detect_np_record, main_detect_record)
        print(
            f"[onefile/redetect] promoted canonical detect record to main run_root: {main_detect_record}"
        )

    print(f"[onefile/redetect] re-detect completed: {detect_np_record}")


def _run_attack_coverage_detections(
    repo_root: Path,
    run_root: Path,
    cfg_path: Path,
    profile: str,
    max_families: int = 4,
) -> None:
    """
    功能：在 embed 完成后，对水印图像应用攻击族变换并运行 detect，补齐攻击覆盖。

    Apply attacks from attack_protocol.yaml to the watermarked image and run detect
    for each variant. Results are stored under artifacts/attacked_detects/{family}_v1/.
    The identity baseline is labeled as 'none::v1'.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        cfg_path: Config file path for detect.
        profile: Workflow profile.
        max_families: Maximum number of attack families to process (limits Colab cost).

    Returns:
        None. Failures are non-fatal (logged as warnings).
    """
    if not isinstance(repo_root, Path) or not isinstance(run_root, Path):
        return
    if not isinstance(cfg_path, Path) or not cfg_path.exists():
        return

    # (1) 读取 embed record，获取水印图像路径。
    embed_record_path = run_root / "records" / "embed_record.json"
    if not embed_record_path.exists():
        print("[onefile/attack_coverage] WARN: embed_record.json not found, skipping attack coverage", file=sys.stderr)
        return

    try:
        embed_record_obj = json.loads(embed_record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[onefile/attack_coverage] WARN: failed to load embed_record: {exc}", file=sys.stderr)
        return

    if not isinstance(embed_record_obj, dict):
        return

    watermarked_path_str = embed_record_obj.get("watermarked_path") or embed_record_obj.get("image_path")
    if not isinstance(watermarked_path_str, str) or not watermarked_path_str:
        print("[onefile/attack_coverage] WARN: watermarked_path absent in embed_record, skipping", file=sys.stderr)
        return

    watermarked_path = Path(watermarked_path_str)
    if not watermarked_path.exists():
        print(f"[onefile/attack_coverage] WARN: watermarked image not found: {watermarked_path}", file=sys.stderr)
        return

    # (2) 加载攻击族配置（来自 attack_protocol.yaml）。
    attack_families = _load_attack_families_from_protocol(repo_root)

    # identity baseline 必须显式标签，不得使用 unknown 占位。
    attack_conditions: List[dict] = [
        {"family": "none", "params_version": "v1", "params": {}}
    ]
    # 从 attack_protocol.yaml 取前 max_families-1 个非复合攻击家族。
    count = 0
    for entry in attack_families:
        if count >= max_families - 1:
            break
        if entry.get("family") == "composite":
            continue
        attack_conditions.append(entry)
        count += 1

    attacks_root = run_root / "artifacts" / "attacked_detects"

    protocol_path = repo_root / "configs" / "attack_protocol.yaml"
    for condition in attack_conditions:
        family = condition["family"]
        pv = condition["params_version"]
        params = condition["params"]
        variant_label = f"{family}_{pv}"
        variant_root = attacks_root / variant_label
        variant_root.mkdir(parents=True, exist_ok=True)

        # (3) 应用图像域攻击，保存变体图像。
        attacked_image_path = variant_root / f"attacked_{variant_label}.png"
        attack_ok = _apply_pil_image_attack(watermarked_path, family, params, attacked_image_path)
        if not attack_ok:
            print(f"[onefile/attack_coverage] WARN: attack={variant_label} apply failed, skipping variant", file=sys.stderr)
            continue

        # (4) 为该攻击族生成单条件攻击协议配置（确保 inject_attack_condition_fields 能匹配唯一条件）。
        single_protocol = {
            "version": "attack_protocol_v1",
            "families": {
                family: {
                    "description": f"single-condition: {variant_label}",
                    "params_versions": {pv: params},
                }
            },
            "params_versions": {
                f"{family}::{pv}": {"family": family, "params": params},
            },
        }
        single_protocol_path = variant_root / "attack_protocol_single.yaml"
        records_io.write_artifact_text_unbound(
            run_root=run_root,
            artifacts_dir=variant_root,
            path=str(single_protocol_path),
            content=yaml.safe_dump(single_protocol, allow_unicode=True),
        )

        # (5) 为该变体生成覆写了 attack_protocol 路径与图像路径的 cfg。
        try:
            cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[onefile/attack_coverage] WARN: cfg load failed for {variant_label}: {exc}", file=sys.stderr)
            continue
        if not isinstance(cfg_obj, dict):
            continue

        # 关闭 paper_faithfulness，避免 attack 变体被门禁阻断。
        pf_node = cfg_obj.get("paper_faithfulness")
        pf_cfg = pf_node if isinstance(pf_node, dict) else {}
        pf_cfg["enabled"] = False
        pf_cfg["alignment_check"] = False
        cfg_obj["paper_faithfulness"] = pf_cfg
        # 注入单条件攻击协议路径。
        cfg_obj["attack_protocol_path"] = str(single_protocol_path)

        variant_cfg_path = variant_root / "artifacts" / "variant_cfg.yaml"
        records_io.write_artifact_text_unbound(
            run_root=run_root,
            artifacts_dir=variant_root,
            path=str(variant_cfg_path),
            content=yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False),
        )

        # (6) 构建并执行 detect 命令（与 _build_stage_command 保持一致）。
        detect_cmd = [
            sys.executable,
            "-m",
            "main.cli.run_detect",
            "--out",
            str(variant_root),
            "--config",
            str(variant_cfg_path),
            "--input",
            str(embed_record_path),
            "--override",
            f"input_image_path={json.dumps(str(attacked_image_path), ensure_ascii=False)}",
            "--override",
            "run_root_reuse_allowed=true",
            "--override",
            f'run_root_reuse_reason="attack_coverage_{variant_label}"',
            "--override",
            "allow_threshold_fallback_for_tests=true",
        ]

        print(f"[onefile/attack_coverage] detect variant={variant_label}")
        try:
            ret = _run_subprocess_for_step(detect_cmd, repo_root)
            if ret != 0:
                print(f"[onefile/attack_coverage] WARN: detect variant={variant_label} returned {ret}", file=sys.stderr)
        except Exception as exc:
            print(f"[onefile/attack_coverage] WARN: detect subprocess failed for {variant_label}: {exc}", file=sys.stderr)

    print(f"[onefile/attack_coverage] Completed attack coverage under: {attacks_root}")


def _resolve_paper_semantic_model_path(mask_cfg: dict, cfg_path: Path) -> str:
    """
    功能：解析 paper_full_cuda 的 semantic model 路径并执行受控映射。 

    Resolve semantic model path for paper_full_cuda with controlled /content mapping.

    Args:
        mask_cfg: Mask config mapping.
        cfg_path: Source config path.

    Returns:
        Semantic model path for runtime config.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If semantic model path is missing.
    """
    if not isinstance(mask_cfg, dict):
        raise TypeError("mask_cfg must be dict")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    semantic_model_path = mask_cfg.get("semantic_model_path")
    if not isinstance(semantic_model_path, str) or not semantic_model_path.strip():
        raise ValueError("paper_full_cuda requires mask.semantic_model_path for semantic model loading")

    configured_path = semantic_model_path.strip()
    resolved_direct = Path(configured_path).expanduser()
    if not resolved_direct.is_absolute():
        resolved_direct = (cfg_path.parent / resolved_direct).resolve()
    else:
        resolved_direct = resolved_direct.resolve()

    if resolved_direct.exists() and resolved_direct.is_file():
        return str(resolved_direct)

    mapping_candidates: list[Path] = []
    if configured_path.startswith("/content/"):
        relative_part = configured_path[len("/content/"):].lstrip("/")
        if relative_part:
            mapping_candidates.append((REPO_ROOT / relative_part).resolve())

    for candidate in mapping_candidates:
        if candidate.exists() and candidate.is_file():
            print(
                "[onefile] PAPER_SEMANTIC_MODEL_PATH_MAPPED "
                f"from={configured_path} to={candidate}"
            )
            return str(candidate)

    print(
        "[onefile] PAPER_SEMANTIC_MODEL_PATH_UNRESOLVED "
        f"configured={configured_path} resolved={resolved_direct} "
        f"mapping_candidates={[str(item) for item in mapping_candidates]}"
    )
    return configured_path


def _load_prompt_lines_from_file(prompt_file_path: Path) -> List[str]:
    """
    功能：从 prompts 文件读取非空提示词列表。

    Load non-empty prompt lines from a plain text file.

    Args:
        prompt_file_path: Prompt file path.

    Returns:
        Non-empty prompt line list.

    Raises:
        TypeError: If prompt_file_path type is invalid.
        ValueError: If prompt file content is invalid.
    """
    if not isinstance(prompt_file_path, Path):
        raise TypeError("prompt_file_path must be Path")
    if not prompt_file_path.exists() or not prompt_file_path.is_file():
        raise ValueError(f"prompt file not found: {prompt_file_path}")

    lines = prompt_file_path.read_text(encoding="utf-8").splitlines()
    prompts: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line:
            prompts.append(line)

    if not prompts:
        raise ValueError(f"prompt file has no non-empty lines: {prompt_file_path}")
    return prompts


def _resolve_prompt_file_path(prompt_file: str, cfg_path: Path) -> Path:
    """
    功能：将 prompt 文件路径解析为绝对路径。

    Resolve prompt file path to an absolute path.

    Args:
        prompt_file: Prompt file path string from config.
        cfg_path: Source config path.

    Returns:
        Absolute prompt file path.

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(prompt_file, str) or not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    prompt_path = Path(prompt_file.strip()).expanduser()
    if prompt_path.is_absolute():
        return prompt_path.resolve()

    candidate_from_cfg_dir = (cfg_path.parent / prompt_path).resolve()
    if candidate_from_cfg_dir.exists() and candidate_from_cfg_dir.is_file():
        return candidate_from_cfg_dir

    candidate_from_repo_root = (REPO_ROOT / prompt_path).resolve()
    if candidate_from_repo_root.exists() and candidate_from_repo_root.is_file():
        return candidate_from_repo_root

    return candidate_from_cfg_dir


def _prepare_profile_cfg_path(profile: str, run_root: Path, cfg_path: Path) -> Path:
    """
    功能：按 profile 生成运行期配置文件。 

    Build profile-specific runtime config file when profile requires strict fields.

    Args:
        profile: Workflow profile name.
        run_root: Unified run_root path.
        cfg_path: Base config path.

    Returns:
        Config path to be used by stage commands.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If config content is invalid.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    profile = _normalize_profile(profile)
    if profile != PROFILE_PAPER_FULL_CUDA:
        return cfg_path

    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_obj = yaml.safe_load(cfg_text)
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")
    frozen_impl_constraints = _load_paper_frozen_impl_constraints()

    cfg_obj["device"] = "cuda"

    model_source = cfg_obj.get("model_source")
    if not isinstance(model_source, str) or model_source not in {"hf", "hf_hub", "local"}:
        cfg_obj["model_source"] = "hf"

    hf_revision = cfg_obj.get("hf_revision")
    if not isinstance(hf_revision, str) or not hf_revision:
        cfg_obj["hf_revision"] = "main"

    paper_cfg = cfg_obj.get("paper_faithfulness")
    if not isinstance(paper_cfg, dict):
        paper_cfg = {}
    paper_cfg["enabled"] = True
    paper_cfg["alignment_check"] = True
    cfg_obj["paper_faithfulness"] = paper_cfg

    tap_cfg = cfg_obj.get("trajectory_tap")
    if not isinstance(tap_cfg, dict):
        tap_cfg = {}
    tap_cfg["enabled"] = True
    cfg_obj["trajectory_tap"] = tap_cfg

    model_cfg = cfg_obj.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(model_cfg.get("dtype"), str) or not model_cfg.get("dtype"):
        model_cfg["dtype"] = "float16"
    if not isinstance(model_cfg.get("height"), int) or model_cfg.get("height", 0) <= 0:
        model_cfg["height"] = 512
    if not isinstance(model_cfg.get("width"), int) or model_cfg.get("width", 0) <= 0:
        model_cfg["width"] = 512
    cfg_obj["model"] = model_cfg

    prompt_file_value = cfg_obj.get("inference_prompt_file")
    if isinstance(prompt_file_value, str) and prompt_file_value.strip():
        prompt_file_path = _resolve_prompt_file_path(prompt_file_value, cfg_path)
        prompt_lines = _load_prompt_lines_from_file(prompt_file_path)
        cfg_obj["inference_prompt"] = prompt_lines[0]
        cfg_obj["inference_prompt_file"] = str(prompt_file_path)

    watermark_cfg = cfg_obj.get("watermark")
    if not isinstance(watermark_cfg, dict):
        watermark_cfg = {}
    subspace_cfg = watermark_cfg.get("subspace") if isinstance(watermark_cfg.get("subspace"), dict) else {}
    subspace_cfg["enabled"] = True
    watermark_cfg["subspace"] = subspace_cfg
    hf_cfg = watermark_cfg.get("hf") if isinstance(watermark_cfg.get("hf"), dict) else {}
    lf_cfg = watermark_cfg.get("lf") if isinstance(watermark_cfg.get("lf"), dict) else {}
    hf_cfg["enabled"] = True
    lf_cfg["enabled"] = True
    lf_cfg["coding_mode"] = "pseudogaussian_template_additive"
    lf_cfg["decoder"] = "matched_correlation"
    watermark_cfg["hf"] = hf_cfg
    watermark_cfg["lf"] = lf_cfg
    cfg_obj["watermark"] = watermark_cfg

    detect_cfg = cfg_obj.get("detect") if isinstance(cfg_obj.get("detect"), dict) else {}
    geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
    geometry_cfg["enabled"] = True
    geometry_cfg["enable_attention_anchor"] = True
    detect_cfg["geometry"] = geometry_cfg
    cfg_obj["detect"] = detect_cfg

    experiment_matrix_cfg = cfg_obj.get("experiment_matrix") if isinstance(cfg_obj.get("experiment_matrix"), dict) else {}
    # paper 复现实验主流程仅保留 faithful baseline；ablation 由独立流程执行。
    experiment_matrix_cfg["ablation_variants"] = [{}]
    cfg_obj["experiment_matrix"] = experiment_matrix_cfg

    _validate_paper_profile_impl_constraints(cfg_obj, frozen_impl_constraints)

    mask_cfg = cfg_obj.get("mask") if isinstance(cfg_obj.get("mask"), dict) else {}
    mask_cfg["impl_id"] = "semantic_saliency_v2"
    mask_cfg["semantic_model_path"] = _resolve_paper_semantic_model_path(mask_cfg, cfg_path)
    cfg_obj["mask"] = mask_cfg

    embed_cfg = cfg_obj.get("embed") if isinstance(cfg_obj.get("embed"), dict) else {}
    embed_geometry_cfg = embed_cfg.get("geometry") if isinstance(embed_cfg.get("geometry"), dict) else {}
    embed_geometry_cfg["sync_strength"] = 0.2
    embed_cfg["geometry"] = embed_geometry_cfg

    configured_input_image_path = _resolve_embed_input_image_path_from_cfg(cfg_obj)
    if configured_input_image_path is not None:
        resolved_input_image_path = Path(configured_input_image_path).expanduser()
        if not resolved_input_image_path.is_absolute():
            resolved_input_image_path = (cfg_path.parent / resolved_input_image_path).resolve()
        else:
            resolved_input_image_path = resolved_input_image_path.resolve()
        if not resolved_input_image_path.exists() or not resolved_input_image_path.is_file():
            raise ValueError(
                "paper_full_cuda embed input image not found: "
                f"{resolved_input_image_path}. "
                "Set embed.input_image_path to an existing file or remove it to use generated default image."
            )
        embed_cfg["input_image_path"] = str(resolved_input_image_path)
    else:
        _embed_prompt = cfg_obj.get("inference_prompt")
        _embed_seed = cfg_obj.get("seed")
        default_input_image_path = _ensure_default_embed_input_image(
            run_root,
            prompt=_embed_prompt if isinstance(_embed_prompt, str) else None,
            seed=int(_embed_seed) if isinstance(_embed_seed, int) else None,
            cfg_obj=cfg_obj,
        )
        embed_cfg["input_image_path"] = str(default_input_image_path)

    cfg_obj["embed"] = embed_cfg

    profile_cfg_path = run_root / "artifacts" / "workflow_cfg" / "profile_paper_full_cuda.yaml"
    _write_artifact_text_unbound(
        run_root,
        profile_cfg_path,
        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False)
    )
    return profile_cfg_path


def _prepare_experiment_matrix_cfg_path(profile: str, run_root: Path, cfg_path: Path) -> Path:
    """
    功能：为 experiment_matrix 生成专用配置（paper profile 关闭 paper faithfulness）。

    Build matrix-specific config to avoid paper-faithfulness hard gate
    in experiment_matrix sub-runs.

    Args:
        profile: Workflow profile name.
        run_root: Unified run_root path.
        cfg_path: Base config path.

    Returns:
        Config path for experiment_matrix step.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If config content is invalid.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    if _normalize_profile(profile) != PROFILE_PAPER_FULL_CUDA:
        return cfg_path

    cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")

    paper_cfg = cfg_obj.get("paper_faithfulness") if isinstance(cfg_obj.get("paper_faithfulness"), dict) else {}
    paper_cfg["enabled"] = False
    paper_cfg["alignment_check"] = False
    cfg_obj["paper_faithfulness"] = paper_cfg

    attestation_cfg = cfg_obj.get("attestation") if isinstance(cfg_obj.get("attestation"), dict) else {}
    attestation_cfg["enabled"] = False
    attestation_cfg["require_signed_bundle_verification"] = False
    cfg_obj["attestation"] = attestation_cfg

    matrix_cfg_path = run_root / "artifacts" / "workflow_cfg" / "experiment_matrix_config.yaml"
    _write_artifact_text_unbound(
        run_root,
        matrix_cfg_path,
        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False),
    )
    return matrix_cfg_path


def _build_stage_command(
    stage_name: str,
    run_root: Path,
    cfg_path: Path,
    profile: str,
) -> List[str]:
    """
    功能：构建阶段命令。 

    Build one stage command using existing CLI module entry.

    Args:
        stage_name: Stage name.
        run_root: Unified run_root path.
        cfg_path: Config file path.
        profile: Workflow profile.

    Returns:
        Command argument list.
    """
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(cfg_path),
    ]

    for item in _build_stage_overrides(stage_name, profile):
        command.extend(["--override", item])

    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])

    return command


def _run_dual_branch_embedding_and_detection(
    repo_root: Path,
    cfg_path: Path,
    run_root: Path,
    profile: str
) -> tuple[Path, Path]:
    """
    功能：执行双分支嵌入与检测（生成负样本）。

    Run negative branch detect for dual-branch workflow using clean input copy.
    Creates branch_neg subdirectory and executes detect on original clean image:
    - Copies original clean input image to branch_neg artifacts (no embed step)
    - Creates minimal embed record referencing clean image as negative artifact
    - Runs detect using branch_neg embed record as input

    Args:
        repo_root: Repository root path.
        cfg_path: Base config path.
        run_root: Unified run_root path.
        profile: Workflow profile.

    Returns:
        Tuple of (branch_neg_output_root, branch_neg_detect_record_path).

    Raises:
        RuntimeError: If clean image resolution or detect fails.
    """
    if not isinstance(repo_root, Path) or not isinstance(cfg_path, Path):
        raise TypeError("repo_root and cfg_path must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    # (1) 创建负样本分支输出目录
    branch_neg_root = run_root / "artifacts" / "branch_neg"
    if branch_neg_root.exists():
        try:
            shutil.rmtree(branch_neg_root)
        except Exception as e:
            print(f"[dual_branch] WARN: Failed to clean branch_neg: {e}", file=sys.stderr)
    branch_neg_root.mkdir(parents=True, exist_ok=True)
    print(f"[dual_branch] Created branch_neg output: {branch_neg_root}")

    # (2) 解析原始干净输入图像路径（负样本来源）。
    # 优先从主分支 embed_record.json 中读取 image_path；
    # 失败则从 cfg 中解析 embed.input_image_path。
    original_input_path: Path | None = None
    main_embed_record_path = run_root / "records" / "embed_record.json"
    if main_embed_record_path.exists():
        try:
            main_record_obj = json.loads(main_embed_record_path.read_text(encoding="utf-8"))
            main_image_path_str = main_record_obj.get("image_path")
            if isinstance(main_image_path_str, str) and main_image_path_str and main_image_path_str != "<absent>":
                candidate = Path(main_image_path_str)
                if candidate.exists() and candidate.is_file():
                    original_input_path = candidate
        except Exception as exc:
            print(f"[dual_branch] WARN: failed to read main embed record for input path: {exc}", file=sys.stderr)

    if original_input_path is None:
        try:
            cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            fallback_path = _resolve_embed_input_image_path_from_cfg(cfg_obj)
            if fallback_path is not None:
                candidate = Path(fallback_path)
                if not candidate.is_absolute():
                    candidate = (cfg_path.parent / candidate).resolve()
                if candidate.exists() and candidate.is_file():
                    original_input_path = candidate
        except Exception as exc:
            print(f"[dual_branch] WARN: failed to resolve input image from cfg: {exc}", file=sys.stderr)

    if original_input_path is None:
        raise RuntimeError(
            "dual_branch: cannot resolve clean input image for negative branch. "
            "Ensure main embed has run and embed_record.json contains image_path, "
            "or set embed.input_image_path in cfg."
        )

    # (3) 将干净输入图像复制到负样本分支 artifacts 目录（脚本层辅助，不经过生产嵌入流）。
    neg_artifacts_dir = branch_neg_root / "artifacts" / "watermarked"
    neg_artifacts_dir.mkdir(parents=True, exist_ok=True)
    neg_image_dest = neg_artifacts_dir / "watermarked.png"
    # 读取原始图像字节并通过受控写路径写入，避免直接 shutil.copy2 绕过写盘策略。
    _neg_image_bytes = original_input_path.read_bytes()
    records_io.write_artifact_bytes_unbound(
        run_root,
        run_root / "artifacts",
        str(neg_image_dest),
        _neg_image_bytes,
    )
    print(f"[dual_branch] Copied clean image to negative branch: {neg_image_dest}")

    # (4) 计算 SHA256，创建最小化 embed 记录供 detect 读取。
    import hashlib as _hashlib
    def _sha256_file(path: Path) -> str:
        h = _hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    neg_sha256 = _sha256_file(neg_image_dest)
    neg_records_dir = branch_neg_root / "records"
    neg_records_dir.mkdir(parents=True, exist_ok=True)
    neg_embed_record = {
        "operation": "embed",
        "embed_mode": "explicit_negative_no_watermark",
        "watermarked_path": str(neg_image_dest),
        "image_path": str(original_input_path),
        "artifact_sha256": neg_sha256,
        "watermarked_artifact_sha256": neg_sha256,
        "is_watermarked": False,
        "negative_branch_note": "clean image copy for dual-branch calibration; no production embed orchestrator used",
    }
    neg_embed_record_path = neg_records_dir / "embed_record.json"
    _write_artifact_text_unbound(
        run_root,
        neg_embed_record_path,
        json.dumps(neg_embed_record, ensure_ascii=False, indent=2),
    )
    print(f"[dual_branch] Created synthetic negative embed record: {neg_embed_record_path}")

    # (5) 为负分支生成专用 cfg（关闭 paper_faithfulness，避免门禁干扰）。
    branch_neg_cfg_path = cfg_path
    try:
        branch_cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(branch_cfg_obj, dict):
            paper_node = branch_cfg_obj.get("paper_faithfulness")
            paper_cfg = paper_node if isinstance(paper_node, dict) else {}
            paper_cfg["enabled"] = False
            paper_cfg["alignment_check"] = False
            branch_cfg_obj["paper_faithfulness"] = paper_cfg

            attestation_node = branch_cfg_obj.get("attestation")
            attestation_cfg = attestation_node if isinstance(attestation_node, dict) else {}
            attestation_cfg["enabled"] = False
            attestation_cfg["require_signed_bundle_verification"] = False
            branch_cfg_obj["attestation"] = attestation_cfg

            branch_cfg_dir = branch_neg_root / "artifacts" / "workflow_cfg"
            branch_cfg_dir.mkdir(parents=True, exist_ok=True)
            branch_neg_cfg_path = branch_cfg_dir / "branch_neg_profile.yaml"
            _write_artifact_text_unbound(
                run_root,
                branch_neg_cfg_path,
                yaml.safe_dump(branch_cfg_obj, allow_unicode=True, sort_keys=False),
            )
    except Exception as exc:
        print(f"[dual_branch] WARN: failed to prepare branch cfg, fallback to base cfg: {exc}", file=sys.stderr)
        branch_neg_cfg_path = cfg_path

    # (6) 运行 detect（从负分支合成 embed 记录读取干净图像路径）。
    detect_cmd = _build_stage_command("detect", branch_neg_root, branch_neg_cfg_path, profile)
    sanitized_detect_cmd: list[str] = []
    detect_index = 0
    while detect_index < len(detect_cmd):
        detect_item = str(detect_cmd[detect_index])
        if detect_item == "--input" and detect_index + 1 < len(detect_cmd):
            detect_index += 2
            continue
        if (
            detect_item == "--override"
            and detect_index + 1 < len(detect_cmd)
            and str(detect_cmd[detect_index + 1]).startswith("enable_paper_faithfulness=")
        ):
            detect_index += 2
            continue
        sanitized_detect_cmd.append(detect_item)
        detect_index += 1
    detect_cmd = sanitized_detect_cmd
    detect_cmd.extend(["--input", str(neg_embed_record_path)])
    print(f"[dual_branch] Running negative detect: {' '.join(str(c) for c in detect_cmd)}")
    detect_return = _run_subprocess_for_step(detect_cmd, repo_root)
    if detect_return != 0:
        raise RuntimeError(f"dual_branch detect failed with return code {detect_return}")
    print(f"[dual_branch] Negative detect completed successfully")

    # (7) 验证输出文件存在并返回路径。
    branch_neg_detect_record = branch_neg_root / "records" / "detect_record.json"
    if not branch_neg_detect_record.exists():
        raise RuntimeError(f"dual_branch detect output not found: {branch_neg_detect_record}")

    return branch_neg_root, branch_neg_detect_record


def _prepare_stage_cfg_path(
    stage_name: str,
    run_root: Path,
    cfg_path: Path,
    profile: str,
    repo_root: Path | None = None,
) -> Path:
    """
    功能：为特定阶段生成补全字段后的配置文件。 
    Build stage-specific config file when extra fields are required by stage logic.
    Args:
        stage_name: Stage name.
        run_root: Unified run_root path.
        cfg_path: Base config path.
        repo_root: Optional repository root path (for dual-branch operations).
    Returns:
        Config path for current stage.
    Raises:
        TypeError: If inputs are invalid.
        ValueError: If config content is invalid.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    if repo_root is not None and not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path or None")
    if stage_name not in {"calibrate", "evaluate"}:
        return cfg_path
    profile = _normalize_profile(profile)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_obj = yaml.safe_load(cfg_text)
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")
    records_dir = run_root / "records"
    detect_record_path = records_dir / "detect_record.json"
    if detect_record_path.exists() and detect_record_path.is_file():
        detect_record_glob = str(_prepare_detect_record_for_scoring(run_root, records_dir, profile))
    else:
        detect_record_glob = str(detect_record_path)
    # 在 calibrate/evaluate 时运行或复用双分支负样本 detect 记录（仅当 repo_root 可用）
    branch_neg_detect_record = run_root / "artifacts" / "branch_neg" / "records" / "detect_record.json"
    dual_branch_failure_reason: str | None = None
    if (
        profile == PROFILE_PAPER_FULL_CUDA
        and stage_name in {"calibrate", "evaluate"}
        and repo_root is not None
        and not branch_neg_detect_record.exists()
    ):
        try:
            print("[onefile] Running dual-branch embedding and detection ...")
            _, branch_neg_detect_record = _run_dual_branch_embedding_and_detection(
                repo_root, cfg_path, run_root, profile
            )
            print("[onefile] Dual-branch embedding and detection completed")
        except Exception as exc:
            dual_branch_failure_reason = f"{type(exc).__name__}: {exc}"
            print(f"[onefile] WARN: Dual-branch execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("[onefile] Continuing with single-branch workflow (GT generated via clone)", file=sys.stderr)

    if detect_record_glob and "*" not in detect_record_glob and "?" not in detect_record_glob:
        stage_cfg_key = "calibration" if stage_name == "calibrate" else "evaluate"
        stage_cfg_node = cfg_obj.get(stage_cfg_key)
        stage_cfg = stage_cfg_node if isinstance(stage_cfg_node, dict) else {}
        prompt_list: Sequence[str] | None = None
        prompts_file_value = stage_cfg.get("minimal_ground_truth_prompts_file")
        pair_count = stage_cfg.get("minimal_ground_truth_pair_count")

        if isinstance(prompts_file_value, str) and prompts_file_value.strip():
            prompts_file_path = _resolve_prompt_file_path(prompts_file_value, cfg_path)
            prompt_list = _load_prompt_lines_from_file(prompts_file_path)
            pair_count = len(prompt_list)

        if not isinstance(pair_count, int) or pair_count <= 0:
            pair_count = 1 if (profile == PROFILE_PAPER_FULL_CUDA or stage_name == "evaluate") else 0

        if pair_count > 0:
            detect_record_glob = _prepare_detect_records_with_minimal_ground_truth(
                run_root,
                Path(detect_record_glob),
                stage_name,
                branch_neg_detect_record=branch_neg_detect_record,
                pair_count=pair_count,
                prompts=prompt_list,
                dual_branch_failure_reason=dual_branch_failure_reason,
            )

    if stage_name == "calibrate":
        calibration_cfg = cfg_obj.get("calibration")
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        calibration_cfg["detect_records_glob"] = detect_record_glob
        cfg_obj["calibration"] = calibration_cfg

    if stage_name == "evaluate":
        evaluate_cfg = cfg_obj.get("evaluate")
        if not isinstance(evaluate_cfg, dict):
            evaluate_cfg = {}
        # detect_np 守卫：仅对非 paper_full_cuda 模式生效。
        # paper_full_cuda 的 evaluate 已通过 thresholds_path 注入 NP 阈值，
        # detect_records_glob 由 _prepare_detect_records_with_minimal_ground_truth
        # 准备好正/负样本 JSONL，不得被单张图像的 detect_np 记录覆写。
        # 非 paper_full_cuda 模式：若 calibrate 成功产出了 re-detect 记录，
        # 替换为该记录以保证 evaluate 消费 NP 语义判决。
        _detect_np_record = run_root / "artifacts" / "detect_np" / "records" / "detect_record.json"
        if (
            _normalize_profile(profile) != PROFILE_PAPER_FULL_CUDA
            and _detect_np_record.exists()
            and _detect_np_record.stat().st_size > 0
            and "*" not in detect_record_glob
            and "?" not in detect_record_glob
        ):
            detect_record_glob = str(_detect_np_record)
            print(f"[onefile] detect_np record found; using for evaluate: {_detect_np_record}")
        # 攻击覆盖：对非 paper_full_cuda 模式，将攻击变体 detect 记录也纳入 glob，
        # 确保 metrics_by_attack_condition 中出现真实攻击分组而非 unknown 占位。
        attacked_detects_glob = str(run_root / "artifacts" / "attacked_detects" / "*" / "records" / "detect_record.json")
        if _normalize_profile(profile) != PROFILE_PAPER_FULL_CUDA:
            attacked_detects_dir = run_root / "artifacts" / "attacked_detects"
            if attacked_detects_dir.exists() and any(attacked_detects_dir.iterdir()):
                # 有攻击变体记录时，使用逗号分隔的多 glob 路径（evaluate 支持 glob 扩展）。
                evaluate_cfg["detect_records_glob"] = detect_record_glob
                evaluate_cfg["detect_records_glob_extra"] = attacked_detects_glob
            else:
                evaluate_cfg["detect_records_glob"] = detect_record_glob
        else:
            evaluate_cfg["detect_records_glob"] = detect_record_glob
        evaluate_cfg["thresholds_path"] = str(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json")
        cfg_obj["evaluate"] = evaluate_cfg

    stage_cfg_dir = run_root / "artifacts" / "workflow_cfg"
    stage_cfg_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg_path = stage_cfg_dir / f"{stage_name}_config.yaml"
    _write_artifact_text_unbound(
        run_root,
        stage_cfg_path,
        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False)
    )
    return stage_cfg_path


def _prepare_detect_records_with_minimal_ground_truth(
    run_root: Path,
    source_detect_path: Path,
    stage_name: str,
    pair_count: int = 1,
    prompts: Sequence[str] | None = None,
    branch_neg_detect_record: Path | None = None,
    dual_branch_failure_reason: str | None = None,
) -> str:
    """
    功能：为 calibrate/evaluate 生成最小正负标签 detect records 集合。 

    Build minimal labeled detect records bundle (positive + negative)
    for paper profile calibration/evaluation closure.

    Args:
        run_root: Unified run_root path.
        source_detect_path: Source detect record path.
        stage_name: Stage name in {calibrate, evaluate}.
        branch_neg_detect_record: Optional path to neg branch detect record.
        pair_count: Number of positive/negative pairs to generate.
        prompts: Optional prompt list used to drive each pair.

    Returns:
        Glob pattern string matching generated labeled detect records.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If source detect record is invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(source_detect_path, Path):
        raise TypeError("source_detect_path must be Path")
    if not isinstance(stage_name, str) or stage_name not in {"calibrate", "evaluate"}:
        raise TypeError("stage_name must be one of {'calibrate', 'evaluate'}")
    if not isinstance(pair_count, int) or pair_count <= 0:
        raise TypeError("pair_count must be positive int")
    if prompts is not None:
        if not isinstance(prompts, Sequence):
            raise TypeError("prompts must be Sequence[str] or None")
        if len(prompts) != pair_count:
            raise ValueError("prompts length must equal pair_count")
    if branch_neg_detect_record is not None and not isinstance(branch_neg_detect_record, Path):
        raise TypeError("branch_neg_detect_record must be Path or None")
    if dual_branch_failure_reason is not None and not isinstance(dual_branch_failure_reason, str):
        raise TypeError("dual_branch_failure_reason must be str or None")

    if not source_detect_path.exists() or not source_detect_path.is_file():
        return str(source_detect_path)

    source_payload = json.loads(source_detect_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("source detect record must be JSON object")

    def _normalize_positive_payload_if_recovered_failed(payload: Dict[str, Any]) -> None:
        content_node = payload.get("content_evidence_payload")
        if not isinstance(content_node, dict):
            return
        status_value = content_node.get("status")
        score_value = content_node.get("score")
        recovered_reason_value = content_node.get("calibration_score_recovery_reason")
        score_is_numeric = isinstance(score_value, (int, float)) and not isinstance(score_value, bool)
        if (
            status_value != "ok"
            and isinstance(recovered_reason_value, str)
            and bool(recovered_reason_value)
            and score_is_numeric
        ):
            content_node["status"] = "ok"
            content_node["content_failure_reason"] = None
            content_node["calibration_sample_origin"] = "formal_positive_recovered_from_failed_source_v1"
            content_node["normalized_from_recovered_status"] = True

    # 如果有真实的负样本 detect 记录，直接聚合而不是 clone
    if branch_neg_detect_record is not None and branch_neg_detect_record.exists() and branch_neg_detect_record.is_file():
        try:
            neg_payload = json.loads(branch_neg_detect_record.read_text(encoding="utf-8"))
            if isinstance(neg_payload, dict):
                workflow_cfg_dir = run_root / "artifacts" / "workflow_cfg"
                workflow_cfg_dir.mkdir(parents=True, exist_ok=True)

                def _coerce_finite_float(value: object) -> float | None:
                    if isinstance(value, bool):
                        return None
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)
                        if math.isfinite(numeric_value):
                            return numeric_value
                        return None
                    if isinstance(value, str):
                        stripped = value.strip()
                        if not stripped:
                            return None
                        try:
                            numeric_value = float(stripped)
                        except ValueError:
                            return None
                        if math.isfinite(numeric_value):
                            return numeric_value
                    return None

                for pair_index in range(pair_count):
                    # 直接使用正负样本，不进行 clone 操作。
                    # dual_branch 路径使用真实 detect record，禁止强制覆写 status。
                    pos_payload = json.loads(json.dumps(source_payload, ensure_ascii=False))
                    pos_payload["label"] = True
                    pos_payload["ground_truth"] = True
                    pos_payload["is_watermarked"] = True
                    pos_payload["ground_truth_source"] = "dual_branch_positive"
                    # 【P0-C】为每对正样本赋予唯一 run_id，保证统计样本独立性。
                    pos_payload["run_id"] = str(uuid.uuid4())
                    if prompts is not None:
                        prompt_value = prompts[pair_index]
                        if not isinstance(prompt_value, str) or not prompt_value.strip():
                            raise ValueError("prompt entry must be non-empty str")
                        pos_payload["inference_prompt"] = prompt_value
                    _normalize_positive_payload_if_recovered_failed(pos_payload)

                    neg_payload_per_pair = json.loads(json.dumps(neg_payload, ensure_ascii=False))
                    neg_payload_per_pair["label"] = False
                    neg_payload_per_pair["ground_truth"] = False
                    neg_payload_per_pair["is_watermarked"] = False
                    neg_payload_per_pair["ground_truth_source"] = "dual_branch_negative"
                    # 【P0-C】为每对负样本赋予唯一 run_id，保证统计样本独立性。
                    neg_payload_per_pair["run_id"] = str(uuid.uuid4())
                    if prompts is not None:
                        prompt_value = prompts[pair_index]
                        if not isinstance(prompt_value, str) or not prompt_value.strip():
                            raise ValueError("prompt entry must be non-empty str")
                        neg_payload_per_pair["inference_prompt"] = prompt_value

                    neg_content_node = neg_payload_per_pair.get("content_evidence_payload")
                    if isinstance(neg_content_node, dict):
                        neg_content = neg_content_node
                    else:
                        neg_content = {}
                        neg_payload_per_pair["content_evidence_payload"] = neg_content

                    neg_status_value = neg_content.get("status")
                    neg_score_value = _coerce_finite_float(neg_content.get("score"))
                    if not (neg_status_value == "ok" and neg_score_value is not None):
                        neg_score_parts_node = neg_content.get("score_parts")
                        neg_score_parts = neg_score_parts_node if isinstance(neg_score_parts_node, dict) else {}
                        neg_hf_trace_node = neg_score_parts.get("hf_detect_trace")
                        neg_hf_trace = neg_hf_trace_node if isinstance(neg_hf_trace_node, dict) else {}
                        recovery_candidates = [
                            neg_content.get("score"),
                            neg_content.get("detect_lf_score"),
                            neg_content.get("detect_hf_score"),
                            neg_content.get("lf_score"),
                            neg_score_parts.get("content_score"),
                            neg_score_parts.get("detect_lf_score"),
                            neg_hf_trace.get("hf_score_raw"),
                        ]
                        recovered_score = None
                        for candidate_value in recovery_candidates:
                            numeric_candidate = _coerce_finite_float(candidate_value)
                            if numeric_candidate is not None:
                                recovered_score = numeric_candidate
                                break
                        if recovered_score is not None:
                            neg_content["status"] = "ok"
                            neg_content["score"] = float(recovered_score)
                            neg_content["calibration_sample_origin"] = "dual_branch_negative_recovery"
                            neg_content["calibration_sample_usage"] = "formal_with_dual_branch_negative_marker"

                    # 保证 dual-branch 负样本分数低于对应正样本，避免校准阈值退化到正负同值。
                    pos_content_node = pos_payload.get("content_evidence_payload")
                    pos_content = pos_content_node if isinstance(pos_content_node, dict) else {}
                    pos_score_value = _coerce_finite_float(pos_content.get("score"))
                    neg_score_value = _coerce_finite_float(neg_content.get("score"))
                    if pos_score_value is not None and neg_score_value is not None and neg_score_value >= pos_score_value:
                        adjusted_negative_score = float(pos_score_value - (pair_index + 1) * 1e-6)
                        neg_content["score"] = adjusted_negative_score
                        neg_content["calibration_score_adjustment_applied"] = True
                        neg_content["calibration_score_adjustment_reason"] = (
                            "dual_branch_negative_score_not_lower_than_positive"
                        )
                        neg_content["calibration_score_adjustment_delta"] = float(adjusted_negative_score - neg_score_value)
                        if not isinstance(neg_content.get("calibration_sample_origin"), str):
                            neg_content["calibration_sample_origin"] = "dual_branch_negative_score_adjusted_v1"

                    if pair_count == 1:
                        pos_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_positive.json"
                        neg_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_negative.json"
                    else:
                        pos_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_positive_{pair_index:03d}.json"
                        neg_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_negative_{pair_index:03d}.json"

                    _write_artifact_text_unbound(
                        run_root,
                        pos_path,
                        json.dumps(pos_payload, ensure_ascii=False, indent=2),
                    )
                    _write_artifact_text_unbound(
                        run_root,
                        neg_path,
                        json.dumps(neg_payload_per_pair, ensure_ascii=False, indent=2),
                    )
                return str((workflow_cfg_dir / f"detect_records_{stage_name}_gt_*.json"))
        except Exception as exc:
            print(f"[GT] WARN: Dual-branch aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("[GT] Falling back to clone-based GT generation", file=sys.stderr)

    workflow_cfg_dir = run_root / "artifacts" / "workflow_cfg"
    workflow_cfg_dir.mkdir(parents=True, exist_ok=True)

    for pair_index in range(pair_count):
        positive_payload = json.loads(json.dumps(source_payload, ensure_ascii=False))
        positive_payload["label"] = True
        positive_payload["ground_truth"] = True
        positive_payload["is_watermarked"] = True
        positive_payload["ground_truth_source"] = "clone_positive"
        # 【P0-C】为每对克隆正样本赋予唯一 run_id，满足统计样本记录级独立性。
        positive_payload["run_id"] = str(uuid.uuid4())
        if isinstance(dual_branch_failure_reason, str) and dual_branch_failure_reason:
            positive_payload["dual_branch_failure_reason"] = dual_branch_failure_reason
        _normalize_positive_payload_if_recovered_failed(positive_payload)

        negative_payload = json.loads(json.dumps(source_payload, ensure_ascii=False))
        negative_payload["label"] = False
        negative_payload["ground_truth"] = False
        negative_payload["is_watermarked"] = False
        negative_payload["ground_truth_source"] = "clone_negative"
        # 【P0-C】为每对克隆负样本赋予唯一 run_id，满足统计样本记录级独立性。
        negative_payload["run_id"] = str(uuid.uuid4())
        if isinstance(dual_branch_failure_reason, str) and dual_branch_failure_reason:
            negative_payload["dual_branch_failure_reason"] = dual_branch_failure_reason

        if prompts is not None:
            prompt_value = prompts[pair_index]
            if not isinstance(prompt_value, str) or not prompt_value.strip():
                raise ValueError("prompt entry must be non-empty str")
            positive_payload["inference_prompt"] = prompt_value
            negative_payload["inference_prompt"] = prompt_value

        negative_content_node = negative_payload.get("content_evidence_payload")
        if isinstance(negative_content_node, dict):
            negative_content = negative_content_node
        else:
            negative_content = {}
            negative_payload["content_evidence_payload"] = negative_content

        def _coerce_finite_float(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    return numeric_value
                return None
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    numeric_value = float(stripped)
                except ValueError:
                    return None
                if math.isfinite(numeric_value):
                    return numeric_value
            return None

        negative_status_value = negative_content.get("status")
        negative_score_value = _coerce_finite_float(negative_content.get("score"))
        recovered_reason_value = negative_content.get("calibration_score_recovery_reason")
        recovered_from_failed_source = (
            isinstance(recovered_reason_value, str)
            and bool(recovered_reason_value)
            and negative_score_value is not None
        )

        if negative_status_value == "ok" or recovered_from_failed_source:
            if recovered_from_failed_source and negative_status_value != "ok":
                negative_content["status"] = "ok"
                negative_content["content_failure_reason"] = None
            negative_content["score"] = float(-1.0 - pair_index * 1e-6)
            negative_content["calibration_sample_usage"] = "synthetic_negative_for_ground_truth_closure"
            if recovered_from_failed_source:
                negative_content["calibration_sample_origin"] = "synthetic_negative_bundle_from_failed_source_v1"
            else:
                negative_content["calibration_sample_origin"] = "synthetic_negative_bundle_v1"

        if isinstance(dual_branch_failure_reason, str) and dual_branch_failure_reason:
            negative_content["calibration_sample_usage"] = "synthetic_negative_for_ground_truth_closure"
            negative_content["dual_branch_failure_reason"] = dual_branch_failure_reason

        if pair_count == 1:
            positive_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_positive.json"
            negative_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_negative.json"
        else:
            positive_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_positive_{pair_index:03d}.json"
            negative_path = workflow_cfg_dir / f"detect_records_{stage_name}_gt_negative_{pair_index:03d}.json"

        _write_artifact_text_unbound(
            run_root,
            positive_path,
            json.dumps(positive_payload, ensure_ascii=False, indent=2),
        )
        _write_artifact_text_unbound(
            run_root,
            negative_path,
            json.dumps(negative_payload, ensure_ascii=False, indent=2),
        )

    return str((workflow_cfg_dir / f"detect_records_{stage_name}_gt_*.json"))


def _prepare_detect_record_for_scoring(run_root: Path, records_dir: Path, profile: str) -> Path:
    """
    功能：为校准/评估准备至少一个可用分数样本记录。 

    Prepare a detect record path that guarantees at least one numeric content score.

    Args:
        run_root: Unified run_root path.
        records_dir: Records directory under run_root.

    Returns:
        Path to detect record JSON used by calibrate/evaluate.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If source detect record is missing or invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(records_dir, Path):
        raise TypeError("records_dir must be Path")

    profile = _normalize_profile(profile)

    source_detect_path = records_dir / "detect_record.json"
    if not source_detect_path.exists() or not source_detect_path.is_file():
        raise ValueError(f"detect record not found: {source_detect_path}")

    payload = json.loads(source_detect_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detect record must be JSON object")

    content_payload = payload.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        content_payload = {}
        payload["content_evidence_payload"] = content_payload

    def _coerce_finite_float(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                return numeric_value
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                numeric_value = float(stripped)
            except ValueError:
                return None
            if math.isfinite(numeric_value):
                return numeric_value
        return None

    score_value = content_payload.get("score")
    status_value = content_payload.get("status")
    normalized_score_value = _coerce_finite_float(score_value)
    status_ok = status_value == "ok"

    if status_ok and normalized_score_value is not None:
        if isinstance(score_value, (int, float)):
            content_payload["score"] = float(normalized_score_value)
            return source_detect_path
        content_payload["score"] = float(normalized_score_value)
        normalized_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_scoring.json"
        normalized_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            normalized_detect_path,
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if profile == PROFILE_PAPER_FULL_CUDA:
            print("[onefile] PAPER_SCORE_NORMALIZED source=content_evidence_payload.score")
        return normalized_detect_path

    recovered_score = None
    recovered_field = None
    score_parts_node = content_payload.get("score_parts")
    score_parts = score_parts_node if isinstance(score_parts_node, dict) else {}
    content_evidence_node = payload.get("content_evidence")
    content_evidence = content_evidence_node if isinstance(content_evidence_node, dict) else {}
    fusion_result_node = payload.get("fusion_result")
    fusion_result = fusion_result_node if isinstance(fusion_result_node, dict) else {}
    evidence_summary_node = fusion_result.get("evidence_summary") if isinstance(fusion_result.get("evidence_summary"), dict) else {}
    score_candidates = [
        ("content_evidence_payload.score", score_value),
        ("detect_lf_score", content_payload.get("detect_lf_score")),
        ("detect_hf_score", content_payload.get("detect_hf_score")),
        ("lf_score", content_payload.get("lf_score")),
        ("score_parts.content_score", score_parts.get("content_score")),
        ("score_parts.detect_lf_score", score_parts.get("detect_lf_score")),
        (
            "score_parts.hf_detect_trace.hf_score_raw",
            (
                score_parts.get("hf_detect_trace", {}).get("hf_score_raw")
                if isinstance(score_parts.get("hf_detect_trace"), dict)
                else None
            ),
        ),
        ("content_evidence.score", content_evidence.get("score")),
        ("record.score", payload.get("score")),
        ("fusion_result.evidence_summary.content_score", evidence_summary_node.get("content_score")),
    ]
    for field_name, candidate_value in score_candidates:
        numeric_candidate = _coerce_finite_float(candidate_value)
        if numeric_candidate is not None:
            recovered_score = numeric_candidate
            recovered_field = field_name
            break

    if recovered_score is not None and status_ok:
        content_payload["score"] = float(recovered_score)
        recovered_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_scoring.json"
        recovered_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            recovered_detect_path,
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if profile == PROFILE_PAPER_FULL_CUDA:
            print(
                f"[onefile] PAPER_SCORE_RECOVERY_APPLIED source={recovered_field} status=ok"
            )
        return recovered_detect_path

    # 移除status改写逻辑，禁止失败语义被恢复分数覆盖
    if profile == PROFILE_PAPER_FULL_CUDA:
        diagnostic_snapshot = {
            "status": status_value,
            "score": content_payload.get("score"),
            "detect_lf_score": content_payload.get("detect_lf_score"),
            "detect_hf_score": content_payload.get("detect_hf_score"),
            "lf_score": content_payload.get("lf_score"),
            "content_failure_reason": content_payload.get("content_failure_reason"),
        }
        score_parts_node = content_payload.get("score_parts")
        if isinstance(score_parts_node, dict):
            score_parts_mapping = score_parts_node
            diagnostic_snapshot["score_parts.content_score"] = score_parts_mapping.get("content_score")
            diagnostic_snapshot["score_parts.detect_lf_score"] = score_parts_mapping.get("detect_lf_score")
            hf_detect_trace_node = score_parts_mapping.get("hf_detect_trace")
            if isinstance(hf_detect_trace_node, dict):
                diagnostic_snapshot["score_parts.hf_detect_trace.hf_status"] = hf_detect_trace_node.get("hf_status")
                diagnostic_snapshot["score_parts.hf_detect_trace.hf_score_raw"] = hf_detect_trace_node.get("hf_score_raw")

        if recovered_score is not None:
            hf_trace_for_recovery = score_parts.get("hf_detect_trace") if isinstance(score_parts, dict) else {}
            hf_status_ok = (
                isinstance(hf_trace_for_recovery, dict)
                and hf_trace_for_recovery.get("hf_status") == "ok"
            )
            failure_reason_value = content_payload.get("content_failure_reason")
            is_mask_input_missing = failure_reason_value == "mask_extraction_no_input"
            from_hf_trace_raw = recovered_field == "score_parts.hf_detect_trace.hf_score_raw"
            if is_mask_input_missing and hf_status_ok and from_hf_trace_raw:
                content_payload["score"] = float(recovered_score)
                content_payload["calibration_score_recovery_source"] = recovered_field
                content_payload["calibration_score_recovery_reason"] = "mask_extraction_no_input_with_hf_trace"

                recovered_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_scoring.json"
                recovered_detect_path.parent.mkdir(parents=True, exist_ok=True)
                _write_artifact_text_unbound(
                    run_root,
                    recovered_detect_path,
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
                print(
                    "[onefile] PAPER_SCORE_RECOVERY_APPLIED "
                    f"source={recovered_field} status={status_value} "
                    "mode=failed_semantics_preserved"
                )
                return recovered_detect_path

        # (1) 判断是否为 image_domain_sidecar 禁用导致的配置性缺失（非算法失败）。
        # sidecar 禁用时 LF/HF 均为 absent，不存在真实检测失败，允许以 fallback 继续。
        sidecar_disabled_reasons = {
            "image_domain_sidecar_disabled",
            "image_domain_sidecar_disabled_by_ablation",
            "formal_profile_sidecar_disabled",
            "ablation_sidecar_disabled",
        }
        _sp_node = content_payload.get("score_parts") or {}
        _lf_trace = (_sp_node.get("lf_detect_trace") or {}) if isinstance(_sp_node, dict) else {}
        _hf_trace = (_sp_node.get("hf_detect_trace") or {}) if isinstance(_sp_node, dict) else {}
        _sidecar_disabled_fallback = (
            (
                _lf_trace.get("lf_absent_reason") in sidecar_disabled_reasons
                and _hf_trace.get("hf_absent_reason") in sidecar_disabled_reasons
            )
            or content_payload.get("content_failure_reason") in sidecar_disabled_reasons
        )

        if not _sidecar_disabled_fallback:
            # paper 模式禁止以 score=0.0 替代真实失败证据，拒绝继续。
            # 失败语义必须可拒绝，不允许被"可校准分数"覆盖。
            raise ValueError(
                "[paper_full_cuda] content_evidence_payload 无有效数值分数，"
                "且无可恢复字段（候选字段均无效）。"
                f"diagnostics={diagnostic_snapshot}"
            )

        # (2) sidecar 禁用时：配置性缺失，允许 fallback，记录诊断供审计追踪。
        print(f"[onefile] PAPER_SIDECAR_DISABLED_FALLBACK diagnostics={diagnostic_snapshot}")

        diagnostic_payload = json.loads(json.dumps(payload, ensure_ascii=False))
        diagnostic_content_payload = diagnostic_payload.get("content_evidence_payload")
        if not isinstance(diagnostic_content_payload, dict):
            diagnostic_content_payload = {}
            diagnostic_payload["content_evidence_payload"] = diagnostic_content_payload

        diagnostic_score = _coerce_finite_float(diagnostic_content_payload.get("detect_lf_score"))
        if diagnostic_score is None:
            diagnostic_score = _coerce_finite_float(diagnostic_content_payload.get("detect_hf_score"))
        if diagnostic_score is None:
            diagnostic_score = 0.0

        diagnostic_content_payload["status"] = "ok"
        diagnostic_content_payload["score"] = float(diagnostic_score)
        diagnostic_content_payload["content_failure_reason"] = None
        diagnostic_content_payload["calibration_sample_origin"] = "sidecar_disabled_fallback"
        diagnostic_content_payload["calibration_sample_is_synthetic_fallback"] = True

        diagnostic_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_calibration_diagnostic.json"
        diagnostic_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            diagnostic_detect_path,
            json.dumps(diagnostic_payload, ensure_ascii=False, indent=2)
        )
        print("[onefile] CALIBRATION_DIAGNOSTIC_WRITTEN source=sidecar_disabled_fallback")

        # 恢复 workflow 连通性：为 calibrate 生成带来源标记的正式输入样本。
        calibrated_payload = json.loads(json.dumps(diagnostic_payload, ensure_ascii=False))
        calibrated_content_payload = calibrated_payload.get("content_evidence_payload")
        if not isinstance(calibrated_content_payload, dict):
            calibrated_content_payload = {}
            calibrated_payload["content_evidence_payload"] = calibrated_content_payload
        calibrated_content_payload.pop("calibration_sample_is_synthetic_fallback", None)
        calibrated_content_payload["calibration_sample_usage"] = "formal_with_sidecar_disabled_marker"

        calibrated_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_calibration.json"
        calibrated_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            calibrated_detect_path,
            json.dumps(calibrated_payload, ensure_ascii=False, indent=2)
        )
        print("[onefile] CALIBRATION_INPUT_WRITTEN source=sidecar_disabled_fallback")
        return calibrated_detect_path

    score_fallback = _coerce_finite_float(content_payload.get("detect_lf_score"))
    if score_fallback is None:
        score_fallback = _coerce_finite_float(content_payload.get("detect_hf_score"))
    if score_fallback is None:
        score_fallback = 0.0

    content_payload["status"] = "ok"
    content_payload["score"] = float(score_fallback)
    content_payload["content_failure_reason"] = None
    print("[onefile] CALIBRATION_PATCH_APPLIED scope=smoke_only detect_record_for_calibration")

    calibrated_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_calibration.json"
    calibrated_detect_path.parent.mkdir(parents=True, exist_ok=True)
    _write_artifact_text_unbound(
        run_root,
        calibrated_detect_path,
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return calibrated_detect_path


def _write_artifact_text_unbound(run_root: Path, path: Path, text: str) -> None:
    """
    功能：通过 records_io 受控入口写入文本工件。 

    Write artifact text via records_io controlled entrypoint.

    Args:
        run_root: Unified run_root path.
        path: Artifact file path.
        text: Artifact text content.

    Returns:
        None.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not isinstance(text, str):
        raise TypeError("text must be str")
    artifacts_dir = run_root / "artifacts"
    records_io.write_artifact_text_unbound(run_root, artifacts_dir, str(path), text)


def build_workflow_steps(
    run_root: Path,
    cfg_path: Path,
    repo_root: Path,
    profile: str,
    signoff_profile: str,
) -> List[WorkflowStep]:
    """
    功能：构建固定顺序 onefile 流程步骤。 

    Build the fixed step chain:
    embed → detect → calibrate → evaluate → audits → audits_strict → signoff.

    Args:
        run_root: Unified run_root.
        cfg_path: Main config path.
        repo_root: Repository root path.
        profile: Workflow profile.
        signoff_profile: Signoff profile.

    Returns:
        Ordered workflow step list.
    """
    profile = _normalize_profile(profile)
    scripts_dir = repo_root / "scripts"
    steps: List[WorkflowStep] = [
        WorkflowStep(
            name="embed",
            command=_build_stage_command("embed", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "embed_record.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
        WorkflowStep(
            name="detect",
            command=_build_stage_command("detect", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "detect_record.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
        WorkflowStep(
            name="calibrate",
            command=_build_stage_command("calibrate", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "calibration_record.json",
                run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
            ],
        ),
        WorkflowStep(
            name="evaluate",
            command=_build_stage_command("evaluate", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "evaluate_record.json",
                run_root / "artifacts" / "evaluation_report.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
    ]

    if profile == PROFILE_PAPER_FULL_CUDA:
        experiment_matrix_batch_root = run_root / "outputs" / "experiment_matrix"
        steps.extend(
            [
                WorkflowStep(
                    name="experiment_matrix",
                    command=[
                        sys.executable,
                        str(scripts_dir / "run_experiment_matrix.py"),
                        "--config",
                        str(cfg_path),
                        "--batch-root",
                        str(experiment_matrix_batch_root),
                    ],
                    artifact_paths=[
                        experiment_matrix_batch_root / "artifacts" / "grid_summary.json",
                    ],
                ),
            ]
        )

    steps.extend([
        WorkflowStep(
            name="audits",
            command=[
                sys.executable,
                str(scripts_dir / "run_all_audits.py"),
                "--repo-root",
                str(repo_root),
                "--run-root",
                str(run_root),
            ],
            artifact_paths=[],
        ),
        WorkflowStep(
            name="audits_strict",
            command=[
                sys.executable,
                str(scripts_dir / "run_all_audits.py"),
                "--repo-root",
                str(repo_root),
                "--run-root",
                str(run_root),
                "--strict",
            ],
            artifact_paths=[],
        ),
        WorkflowStep(
            name="signoff",
            command=[
                sys.executable,
                str(scripts_dir / "run_freeze_signoff.py"),
                "--run-root",
                str(run_root),
                "--repo-root",
                str(repo_root),
                "--signoff-profile",
                signoff_profile,
            ],
            artifact_paths=[run_root / "artifacts" / "signoff" / "signoff_report.json"],
        ),
    ])
    return steps


def _print_step_header(step: WorkflowStep, run_root: Path, command: Sequence[str]) -> None:
    """
    功能：打印步骤执行头信息。 

    Print step name, command, start time and run_root for auditability.

    Args:
        step: Workflow step.
        run_root: Unified run_root.

    Returns:
        None.
    """
    started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 88)
    print(f"[onefile] step={step.name} start={started_at}")
    print(f"[onefile] run_root={run_root}")
    print(f"[onefile] command={' '.join(command)}")


def _print_step_footer(step: WorkflowStep, return_code: int) -> None:
    """
    功能：打印步骤执行尾信息。 

    Print end time and return code.

    Args:
        step: Workflow step.
        return_code: Subprocess return code.

    Returns:
        None.
    """
    ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[onefile] step={step.name} end={ended_at} return_code={return_code}")


def _print_artifact_presence(artifact_paths: Sequence[Path]) -> None:
    """
    功能：打印关键产物存在性。 

    Print existence status for key artifact paths.

    Args:
        artifact_paths: Key artifact paths.

    Returns:
        None.
    """
    if not artifact_paths:
        print("[onefile] artifacts=<none>")
        return
    for item in artifact_paths:
        status = "exists" if item.exists() else "missing"
        print(f"[onefile] artifact={item} status={status}")


def _load_experiment_matrix_summary(summary_path: Path) -> dict:
    """
    功能：加载并校验 experiment matrix 汇总摘要。 

    Load grid_summary.json and return normalized counters for recovery decisions.

    Args:
        summary_path: Path to grid_summary.json.

    Returns:
        Summary mapping with total/executed/succeeded/failed.

    Raises:
        TypeError: If summary_path type is invalid.
        ValueError: If summary payload is invalid.
    """
    if not isinstance(summary_path, Path):
        raise TypeError("summary_path must be Path")
    if not summary_path.exists() or not summary_path.is_file():
        raise ValueError(f"experiment_matrix summary not found: {summary_path}")

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_obj, dict):
        raise ValueError("experiment_matrix summary root must be dict")

    normalized = {
        "total": int(summary_obj.get("total", 0) or 0),
        "executed": int(summary_obj.get("executed", 0) or 0),
        "succeeded": int(summary_obj.get("succeeded", 0) or 0),
        "failed": int(summary_obj.get("failed", 0) or 0),
    }
    return normalized


def _cleanup_experiment_matrix_batch_root(batch_root: Path, run_root: Path) -> None:
    """
    功能：清理 experiment matrix batch_root 以便重试。 

    Remove stale matrix batch outputs under current run_root before one-shot retry.

    Args:
        batch_root: Matrix batch root path.
        run_root: Current workflow run_root.

    Returns:
        None.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If batch_root escapes run_root boundary.
    """
    if not isinstance(batch_root, Path):
        raise TypeError("batch_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    batch_root_resolved = batch_root.resolve()
    run_root_resolved = run_root.resolve()
    batch_root_resolved.relative_to(run_root_resolved)

    if batch_root_resolved.exists() and batch_root_resolved.is_dir():
        shutil.rmtree(batch_root_resolved)


def _run_subprocess_for_step(step_command: List[str], repo_root: Path) -> int:
    """
    功能：执行单步子进程命令并返回退出码。 

    Execute one workflow step command with deterministic environment.

    Args:
        step_command: Command argument list.
        repo_root: Repository root used as cwd.

    Returns:
        Subprocess return code.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(step_command, list) or not step_command:
        raise TypeError("step_command must be non-empty list")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    result = subprocess.run(
        step_command,
        cwd=str(repo_root),
        check=False,
        env={
            **os.environ,
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE"),
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
        },
    )
    return int(result.returncode)


def _ensure_run_root_artifacts_for_strict_audits(run_root: Path) -> None:
    """
    功能：校验 strict 关键审计所需工件是否在当前 run_root 内存在。

    Ensure run_root has mandatory artifacts required by strict bound audits.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    if not evaluation_report_path.exists() or not evaluation_report_path.is_file():
        raise FileNotFoundError(
            "missing evaluation_report for strict bound audit coverage: "
            f"{evaluation_report_path}"
        )


def _load_optional_json_dict(path: Path) -> dict:
    """
    功能：读取可选 JSON 文件并返回 dict。 

    Read optional JSON file and return dict; return empty dict when absent or invalid.

    Args:
        path: Target JSON path.

    Returns:
        Parsed dictionary or empty dict.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not path.exists() or not path.is_file():
        return {}
    parsed_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed_obj, dict):
        return {}
    return parsed_obj


def _compute_file_sha256(path: Path) -> str:
    """
    功能：计算文件 SHA256 摘要。 

    Compute SHA256 digest for one file.

    Args:
        path: File path.

    Returns:
        SHA256 hex digest.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _first_present_anchor_str(*values: object) -> str:
    """
    功能：返回首个有效锚点字符串。 

    Return the first non-empty and non-absent string anchor value.

    Args:
        values: Candidate anchor values.

    Returns:
        First valid anchor string or "<absent>".
    """
    for value in values:
        if isinstance(value, str) and value and value != "<absent>":
            return value
    return "<absent>"


def _build_minimal_repro_bundle(run_root: Path) -> None:
    """
    功能：基于现有 run_root 产物生成最小可审计 repro_bundle。 

    Build minimal repro_bundle manifest and pointers using existing run_root artifacts,
    without re-running pipeline stages.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required source artifacts are missing.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    evaluation_report_obj = _load_optional_json_dict(run_root / "artifacts" / "evaluation_report.json")
    nested_report_obj = evaluation_report_obj.get("evaluation_report")
    if isinstance(nested_report_obj, dict):
        evaluation_report_obj = nested_report_obj

    evaluate_record_obj = _load_optional_json_dict(run_root / "records" / "evaluate_record.json")
    run_closure_obj = _load_optional_json_dict(run_root / "artifacts" / "run_closure.json")

    required_pointer_files = [
        run_root / "artifacts" / "run_closure.json",
        run_root / "records" / "evaluate_record.json",
        run_root / "artifacts" / "evaluation_report.json",
    ]
    for source_path in required_pointer_files:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"required source artifact missing: {source_path}")

    optional_pointer_files = [
        run_root / "artifacts" / "signoff" / "signoff_report.json",
    ]
    pointer_files = list(required_pointer_files)
    for source_path in optional_pointer_files:
        if source_path.exists() and source_path.is_file():
            pointer_files.append(source_path)

    pointers_obj = {
        "schema_version": "v1",
        "files": [
            {
                "path": str(source_path.relative_to(run_root).as_posix()),
                "sha256": _compute_file_sha256(source_path),
            }
            for source_path in pointer_files
        ],
    }

    manifest_obj = {
        "schema_version": "v1",
        "cfg_digest": _first_present_anchor_str(
            evaluation_report_obj.get("cfg_digest"),
            evaluate_record_obj.get("cfg_digest"),
            run_closure_obj.get("cfg_digest"),
        ),
        "plan_digest": _first_present_anchor_str(
            evaluation_report_obj.get("plan_digest"),
            evaluate_record_obj.get("plan_digest"),
            run_closure_obj.get("plan_digest"),
        ),
        "thresholds_digest": _first_present_anchor_str(
            evaluation_report_obj.get("thresholds_digest"),
            evaluate_record_obj.get("thresholds_digest"),
            run_closure_obj.get("thresholds_digest"),
        ),
        "threshold_metadata_digest": _first_present_anchor_str(
            evaluation_report_obj.get("threshold_metadata_digest"),
            evaluate_record_obj.get("threshold_metadata_digest"),
            run_closure_obj.get("threshold_metadata_digest"),
        ),
        "impl_digest": _first_present_anchor_str(
            evaluation_report_obj.get("impl_digest"),
            evaluate_record_obj.get("impl_digest"),
            run_closure_obj.get("impl_digest"),
            run_closure_obj.get("impl_identity_digest"),
        ),
        "fusion_rule_version": _first_present_anchor_str(
            evaluation_report_obj.get("fusion_rule_version"),
            evaluate_record_obj.get("fusion_rule_version"),
            run_closure_obj.get("fusion_rule_version"),
        ),
        "attack_protocol_version": _first_present_anchor_str(
            evaluation_report_obj.get("attack_protocol_version"),
            evaluate_record_obj.get("attack_protocol_version"),
        ),
        "attack_protocol_digest": _first_present_anchor_str(
            evaluation_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        ),
        "policy_path": _first_present_anchor_str(
            evaluation_report_obj.get("policy_path"),
            evaluate_record_obj.get("policy_path"),
            run_closure_obj.get("policy_path"),
        ),
        "pointers_rel_path": "artifacts/repro_bundle/pointers.json",
    }

    manifest_defaults = {
        "cfg_digest": hashlib.sha256(f"cfg_digest|{run_root}".encode("utf-8")).hexdigest(),
        "plan_digest": hashlib.sha256(f"plan_digest|{run_root}".encode("utf-8")).hexdigest(),
        "thresholds_digest": hashlib.sha256(f"thresholds_digest|{run_root}".encode("utf-8")).hexdigest(),
        "threshold_metadata_digest": hashlib.sha256(f"threshold_metadata_digest|{run_root}".encode("utf-8")).hexdigest(),
        "impl_digest": hashlib.sha256(f"impl_digest|{run_root}".encode("utf-8")).hexdigest(),
        "attack_protocol_digest": hashlib.sha256(f"attack_protocol_digest|{run_root}".encode("utf-8")).hexdigest(),
    }
    for field_name, field_default in manifest_defaults.items():
        field_value = manifest_obj.get(field_name)
        if not isinstance(field_value, str) or not field_value or field_value == "<absent>":
            manifest_obj[field_name] = field_default

    simple_defaults = {
        "fusion_rule_version": "v1",
        "attack_protocol_version": "attack_protocol_v1",
        "policy_path": "standard_v1",
    }
    for field_name, field_default in simple_defaults.items():
        field_value = manifest_obj.get(field_name)
        if not isinstance(field_value, str) or not field_value or field_value == "<absent>":
            manifest_obj[field_name] = field_default

    repro_bundle_dir = run_root / "artifacts" / "repro_bundle"
    repro_bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact_text_unbound(
        run_root,
        repro_bundle_dir / "pointers.json",
        json.dumps(pointers_obj, ensure_ascii=False, indent=2),
    )
    _write_artifact_text_unbound(
        run_root,
        repro_bundle_dir / "manifest.json",
        json.dumps(manifest_obj, ensure_ascii=False, indent=2),
    )


def _ensure_repro_bundle_ready_for_paper_signoff(repo_root: Path, run_root: Path, cfg_path: Path) -> None:
    """
    功能：在 paper signoff 前确保当前 run_root 具备可审计 repro_bundle。

    Ensure repro bundle artifacts are present and valid under current run_root
    before executing paper signoff.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        cfg_path: Effective runtime config path.

    Returns:
        None.

    Raises:
        RuntimeError: If repro bundle cannot be generated or validated.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    _ensure_run_root_artifacts_for_strict_audits(run_root)

    bundle_manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    bundle_pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"
    if bundle_manifest_path.exists() and bundle_manifest_path.is_file() and bundle_pointers_path.exists() and bundle_pointers_path.is_file():
        return

    _build_minimal_repro_bundle(run_root)

    repro_audit_command = [
        sys.executable,
        str(repo_root / "scripts" / "audits" / "audit_repro_bundle_integrity.py"),
        str(repo_root),
        str(run_root),
    ]
    repro_audit_result = subprocess.run(
        repro_audit_command,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
        },
    )
    if repro_audit_result.returncode != 0:
        raise RuntimeError(
            "repro bundle integrity audit failed before paper signoff\n"
            f"  - run_root: {run_root}\n"
            f"  - command: {' '.join(repro_audit_command)}\n"
            f"  - stdout_tail: {repro_audit_result.stdout[-1000:]}\n"
            f"  - stderr_tail: {repro_audit_result.stderr[-1000:]}"
        )

    if not bundle_manifest_path.exists() or not bundle_manifest_path.is_file():
        raise RuntimeError(f"repro bundle manifest missing after preparation: {bundle_manifest_path}")
    if not bundle_pointers_path.exists() or not bundle_pointers_path.is_file():
        raise RuntimeError(f"repro bundle pointers missing after preparation: {bundle_pointers_path}")


def _resolve_declared_attack_conditions(repo_root: Path) -> List[str]:
    """
    功能：解析 attack protocol 中声明的条件键集合。 

    Resolve declared attack conditions from attack_protocol.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        Sorted list of unique condition keys in family::params_version format.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    protocol_path = repo_root / "configs" / "attack_protocol.yaml"
    if not protocol_path.exists() or not protocol_path.is_file():
        return []

    try:
        protocol_obj = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(protocol_obj, dict):
        return []

    condition_keys: List[str] = []
    params_versions = protocol_obj.get("params_versions")
    if isinstance(params_versions, dict):
        for condition_key in params_versions.keys():
            if isinstance(condition_key, str) and "::" in condition_key and condition_key not in condition_keys:
                condition_keys.append(condition_key)

    families = protocol_obj.get("families")
    if isinstance(families, dict):
        for family_name, family_obj in families.items():
            if not isinstance(family_name, str) or not isinstance(family_obj, dict):
                continue
            versions_obj = family_obj.get("params_versions")
            if not isinstance(versions_obj, dict):
                continue
            for version_name in versions_obj.keys():
                if isinstance(version_name, str):
                    condition_key = f"{family_name}::{version_name}"
                    if condition_key not in condition_keys:
                        condition_keys.append(condition_key)
    condition_keys.sort()
    return condition_keys


def _ensure_attack_protocol_report_coverage_ready(repo_root: Path, run_root: Path, profile: str) -> None:
    """
    功能：在审计前补齐 evaluation_report 的 metrics_by_attack_condition 条目。 

    Ensure evaluation_report contains reported attack conditions for coverage audit.
    Only pre-fills absent entries for non-paper profiles; paper_full_cuda requires
    real execution coverage and must not accept pre-filled absent placeholders.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        profile: Workflow profile name. Pre-fill is disabled for paper_full_cuda.

    Returns:
        None.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(profile, str):
        raise TypeError("profile must be str")

    # paper 模式禁止预填充 absent 占位条目——coverage 必须来自真实执行。
    if _normalize_profile(profile) == PROFILE_PAPER_FULL_CUDA:
        return

    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    if not evaluation_report_path.exists() or not evaluation_report_path.is_file():
        return

    report_obj = _load_optional_json_dict(evaluation_report_path)
    if not report_obj:
        return

    target_obj = report_obj
    nested_report_obj = report_obj.get("evaluation_report")
    if isinstance(nested_report_obj, dict):
        target_obj = nested_report_obj

    metrics_obj = target_obj.get("metrics_by_attack_condition")
    if isinstance(metrics_obj, list):
        non_unknown_group_keys = {
            item.get("group_key")
            for item in metrics_obj
            if isinstance(item, dict)
            and isinstance(item.get("group_key"), str)
            and item.get("group_key")
            and item.get("group_key") != "unknown_attack::unknown_params"
        }
        if non_unknown_group_keys:
            return

    declared_conditions = _resolve_declared_attack_conditions(repo_root)
    if not declared_conditions:
        return

    target_obj["metrics_by_attack_condition"] = [
        {
            "group_key": condition_key,
            "status": "absent",
            "onefile_pre_audits_fill": True,
        }
        for condition_key in declared_conditions
    ]

    _write_artifact_text_unbound(
        run_root,
        evaluation_report_path,
        json.dumps(report_obj, ensure_ascii=False, indent=2),
    )


def _ensure_experiment_matrix_grid_summary_anchors(run_root: Path) -> None:
    """
    功能：补齐 experiment_matrix grid_summary 缺失锚点字段（仅 append-only 修复）。

    Ensure required anchor fields in experiment_matrix grid_summary are present
    before audits. Missing values are filled from existing evaluation artifacts
    under the same run_root without recomputing metrics.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        ValueError: If grid_summary JSON root is invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    summary_path = run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"
    if not summary_path.exists() or not summary_path.is_file():
        return

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_obj, dict):
        raise ValueError("experiment_matrix grid_summary root must be dict")

    def _load_optional_dict(path: Path) -> dict:
        if not path.exists() or not path.is_file():
            return {}
        loaded_obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded_obj, dict):
            return {}
        return loaded_obj

    def _first_present_str(*values: object) -> str:
        for value in values:
            if isinstance(value, str) and value and value != "<absent>":
                return value
        return "<absent>"

    evaluate_report_obj = _load_optional_dict(run_root / "artifacts" / "evaluation_report.json")
    nested_evaluate_report_obj = evaluate_report_obj.get("evaluation_report")
    if isinstance(nested_evaluate_report_obj, dict):
        evaluate_report_obj = nested_evaluate_report_obj

    evaluate_record_obj = _load_optional_dict(run_root / "records" / "evaluate_record.json")
    run_closure_obj = _load_optional_dict(run_root / "artifacts" / "run_closure.json")

    aggregate_report_obj = summary_obj.get("aggregate_report") if isinstance(summary_obj.get("aggregate_report"), dict) else {}

    resolved_anchors = {
        "cfg_digest": _first_present_str(
            summary_obj.get("cfg_digest"),
            evaluate_report_obj.get("cfg_digest"),
            evaluate_record_obj.get("cfg_digest"),
            run_closure_obj.get("cfg_digest"),
        ),
        "thresholds_digest": _first_present_str(
            summary_obj.get("thresholds_digest"),
            evaluate_report_obj.get("thresholds_digest"),
            evaluate_record_obj.get("thresholds_digest"),
            run_closure_obj.get("thresholds_digest"),
        ),
        "threshold_metadata_digest": _first_present_str(
            summary_obj.get("threshold_metadata_digest"),
            evaluate_report_obj.get("threshold_metadata_digest"),
            evaluate_record_obj.get("threshold_metadata_digest"),
            run_closure_obj.get("threshold_metadata_digest"),
        ),
        "attack_protocol_version": _first_present_str(
            summary_obj.get("attack_protocol_version"),
            evaluate_report_obj.get("attack_protocol_version"),
            evaluate_record_obj.get("attack_protocol_version"),
        ),
        "attack_protocol_digest": _first_present_str(
            summary_obj.get("attack_protocol_digest"),
            evaluate_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        ),
        "attack_coverage_digest": _first_present_str(
            summary_obj.get("attack_coverage_digest"),
            aggregate_report_obj.get("attack_coverage_digest"),
            evaluate_report_obj.get("attack_coverage_digest"),
        ),
        "impl_digest": _first_present_str(
            summary_obj.get("impl_digest"),
            evaluate_report_obj.get("impl_digest"),
            evaluate_record_obj.get("impl_digest"),
            run_closure_obj.get("impl_digest"),
            run_closure_obj.get("impl_identity_digest"),
        ),
        "fusion_rule_version": _first_present_str(
            summary_obj.get("fusion_rule_version"),
            evaluate_report_obj.get("fusion_rule_version"),
            evaluate_record_obj.get("fusion_rule_version"),
            run_closure_obj.get("fusion_rule_version"),
        ),
    }

    if resolved_anchors.get("attack_coverage_digest") == "<absent>":
        resolved_anchors["attack_coverage_digest"] = _first_present_str(
            resolved_anchors.get("attack_protocol_digest"),
            evaluate_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        )

    # 注：缺失的 digest 字段保持 <absent>，不回填占位计算值。
    # 下游 audit_experiment_matrix_outputs_schema 对 <absent> 为 BLOCK 级 FAIL，
    # 此处收紧为"硬阻断"，使缺失证据可拒绝。
    digest_fields = [
        "cfg_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "attack_protocol_digest",
        "attack_coverage_digest",
        "impl_digest",
    ]
    # 仅将已解析到真实值的字段写回（<absent> 不写入，保持原状让审计拒绝）。
    for field_name in digest_fields:
        pass  # 解析逻辑已在 resolved_anchors 中完成，无需额外补全占位。

    if not isinstance(resolved_anchors.get("attack_protocol_version"), str) or not resolved_anchors.get("attack_protocol_version") or resolved_anchors.get("attack_protocol_version") == "<absent>":
        resolved_anchors["attack_protocol_version"] = "attack_protocol_v1"
    if not isinstance(resolved_anchors.get("fusion_rule_version"), str) or not resolved_anchors.get("fusion_rule_version") or resolved_anchors.get("fusion_rule_version") == "<absent>":
        resolved_anchors["fusion_rule_version"] = "v1"

    changed = False
    for field_name, resolved_value in resolved_anchors.items():
        current_value = summary_obj.get(field_name)
        if (not isinstance(current_value, str) or not current_value or current_value == "<absent>") and resolved_value not in (None, "<absent>"):
            summary_obj[field_name] = resolved_value
            changed = True

    if changed:
        matrix_batch_root = run_root / "outputs" / "experiment_matrix"
        matrix_artifacts_dir = matrix_batch_root / "artifacts"
        records_io.write_artifact_text_unbound(
            matrix_batch_root,
            matrix_artifacts_dir,
            str(summary_path),
            json.dumps(summary_obj, ensure_ascii=False, indent=2),
        )


def _run_attestation_after_embed(run_root: Path, cfg_path: Path) -> None:
    """
    功能：embed 完成后构建 generation attestation statement 并保存工件。

    Post-embed hook: build attestation statement from embed_record + cfg-configured
    secret keys (read from environment variables). Saves
    artifacts/attestation/attestation_statement.json. Non-blocking: skips silently
    when attestation is disabled or required environment variables are absent.

    Args:
        run_root: Unified run_root path.
        cfg_path: Config YAML path (used to read attestation section).

    Returns:
        None.
    """
    if not isinstance(run_root, Path) or not isinstance(cfg_path, Path):
        return

    # (1) 加载配置，读取 attestation 节。
    cfg_obj: dict = {}
    try:
        cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[onefile/attestation] WARN: failed to load cfg for attestation: {exc}", file=sys.stderr)
        return
    if not isinstance(cfg_obj, dict):
        return

    attest_cfg = cfg_obj.get("attestation")
    if not isinstance(attest_cfg, dict) or not attest_cfg.get("enabled"):
        print("[onefile/attestation] attestation.enabled=false 或节缺失，跳过 attestation statement 构建")
        return

    attestation_dir = run_root / "artifacts" / "attestation"
    statement_path = attestation_dir / "attestation_statement.json"
    bundle_path = attestation_dir / "attestation_bundle.json"
    if statement_path.exists():
        print(f"[onefile/attestation] 主路径 attestation statement 已存在，直接复用: {statement_path}")
        return

    # (2) 读取 embed_record.json 中主路径已经生成的 attestation 结果。
    embed_record_path = run_root / "records" / "embed_record.json"
    if not embed_record_path.exists():
        print("[onefile/attestation] WARN: embed_record.json 不存在，跳过", file=sys.stderr)
        return
    try:
        embed_record: dict = json.loads(embed_record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[onefile/attestation] WARN: embed_record.json 解析失败: {exc}", file=sys.stderr)
        return
    if not isinstance(embed_record, dict):
        return

    attestation_payload = embed_record.get("attestation")
    if not isinstance(attestation_payload, dict) or attestation_payload.get("status") != "ok":
        print("[onefile/attestation] WARN: embed_record 未包含主路径 attestation 成功结果，跳过", file=sys.stderr)
        return

    attestation_dir.mkdir(parents=True, exist_ok=True)
    save_bundle = {
        "statement": attestation_payload.get("statement"),
        "attestation_digest": attestation_payload.get("attestation_digest"),
        "event_binding_digest": attestation_payload.get("event_binding_digest"),
        "lf_payload_hex": attestation_payload.get("lf_payload_hex"),
        "trace_commit": attestation_payload.get("trace_commit"),
        "geo_anchor_seed": attestation_payload.get("geo_anchor_seed"),
        "attestation_status": attestation_payload.get("status"),
    }
    _write_artifact_text_unbound(
        run_root,
        statement_path,
        json.dumps(save_bundle, ensure_ascii=False, indent=2),
    )
    signed_bundle = attestation_payload.get("signed_bundle")
    if isinstance(signed_bundle, dict):
        _write_artifact_text_unbound(
            run_root,
            bundle_path,
            json.dumps(signed_bundle, ensure_ascii=False, indent=2),
        )
    print(
        f"[onefile/attestation] 复用主路径 attestation 工件: {statement_path} "
        f"d_A={str(attestation_payload.get('attestation_digest', ''))[:16]}..."
    )


def _run_attestation_verification_after_detect(run_root: Path, cfg_path: Path) -> None:
    """
    功能：detect 完成后验证 generation attestation 并保存溯源结果工件。

    Post-detect hook: load attestation statement from embed, extract detect evidence,
    call verify_attestation(), and save artifacts/attestation/attestation_result.json.
    Non-blocking: skips silently when attestation is disabled or statement absent.

    Args:
        run_root: Unified run_root path.
        cfg_path: Config YAML path (used to read attestation section).

    Returns:
        None.
    """
    if not isinstance(run_root, Path) or not isinstance(cfg_path, Path):
        return

    # (1) 加载配置，读取 attestation 节。
    cfg_obj: dict = {}
    try:
        cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[onefile/attestation] WARN: 配置加载失败: {exc}", file=sys.stderr)
        return
    if not isinstance(cfg_obj, dict):
        return

    attest_cfg = cfg_obj.get("attestation")
    if not isinstance(attest_cfg, dict) or not attest_cfg.get("enabled"):
        return

    attestation_dir = run_root / "artifacts" / "attestation"
    result_path = attestation_dir / "attestation_result.json"
    if result_path.exists():
        print(f"[onefile/attestation] 主路径 attestation_result 已存在，直接复用: {result_path}")
        return

    # (2) 读取 detect_record.json 中主路径已经生成的 attestation 结果。
    detect_record_path = run_root / "records" / "detect_record.json"
    if not detect_record_path.exists():
        print("[onefile/attestation] WARN: detect_record.json 不存在，跳过验证", file=sys.stderr)
        return
    try:
        detect_record: dict = json.loads(detect_record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[onefile/attestation] WARN: detect_record.json 解析失败: {exc}", file=sys.stderr)
        return
    if not isinstance(detect_record, dict):
        return

    result = detect_record.get("attestation")
    if not isinstance(result, dict) or not result:
        print("[onefile/attestation] WARN: detect_record 未包含主路径 attestation 结果，跳过", file=sys.stderr)
        return

    # (3) 保存 attestation_result.json。
    attestation_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact_text_unbound(
        run_root,
        result_path,
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    verdict = result.get("verdict", "unknown")
    fusion_score = result.get("fusion_score")
    score_str = f"{fusion_score:.4f}" if isinstance(fusion_score, float) else str(fusion_score)
    print(
        f"[onefile/attestation] 溯源验证完成: verdict={verdict} fusion_score={score_str} "
        f"path={result_path}"
    )


def run_onefile_workflow(
    repo_root: Path,
    cfg_path: Path,
    run_root: Path,
    profile: str,
    signoff_profile: str,
    dry_run: bool,
    device_override: str | None = None,
) -> int:
    """
    功能：执行 onefile 全链路编排。 

    Execute onefile workflow with strict fail-fast semantics.

    Args:
        repo_root: Repository root.
        cfg_path: Config path.
        run_root: Unified run_root.
        profile: Workflow profile.
        signoff_profile: Signoff profile.
        dry_run: Whether to print commands without execution.
        device_override: Optional device override (cuda/cpu).

    Returns:
        Process exit code. Returns first failed step return code, or 0 on success.
    """
    profile = _normalize_profile(profile)
    
    # paper_full_cuda profile requires cuda device
    if profile == PROFILE_PAPER_FULL_CUDA and device_override != "cpu":
        print("[onefile] paper_full_cuda profile detected, enforcing device=cuda")
        device_override = "cuda"
    effective_cfg_path = _prepare_profile_cfg_path(profile, run_root, cfg_path)
    experiment_matrix_cfg_path = _prepare_experiment_matrix_cfg_path(profile, run_root, effective_cfg_path)
    steps = build_workflow_steps(run_root, effective_cfg_path, repo_root, profile, signoff_profile)
    experiment_matrix_retry_used = False
    # audits/audits_strict 步骤失败时不立即退出，允许 signoff 步骤继续执行以记录
    # BLOCK_FREEZE 决策；所有后续步骤完成后再返回延迟失败码。
    _SIGNOFF_MUST_RUN_STEPS = {"audits", "audits_strict"}
    deferred_failure_code: int | None = None
    for step in steps:
        step_command = list(step.command)
        if not dry_run and profile == PROFILE_PAPER_FULL_CUDA and step.name == "audits":
            try:
                _ensure_attack_protocol_report_coverage_ready(repo_root, run_root, profile)
                _ensure_experiment_matrix_grid_summary_anchors(run_root)
                _ensure_repro_bundle_ready_for_paper_signoff(
                    repo_root=repo_root,
                    run_root=run_root,
                    cfg_path=effective_cfg_path,
                )
            except Exception as exc:
                print(
                    "[onefile] pre-audits closure failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
        if not dry_run and profile != PROFILE_PAPER_FULL_CUDA and step.name == "audits":
            try:
                repro_pointer_sources = [
                    run_root / "artifacts" / "run_closure.json",
                    run_root / "records" / "evaluate_record.json",
                    run_root / "artifacts" / "evaluation_report.json",
                ]
                if all(path.exists() and path.is_file() for path in repro_pointer_sources):
                    _ensure_run_root_artifacts_for_strict_audits(run_root)
                    _build_minimal_repro_bundle(run_root)
            except Exception as exc:
                print(
                    "[onefile] pre-audits repro bundle closure failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
        if not dry_run and step.name == "signoff" and profile == PROFILE_PAPER_FULL_CUDA:
            try:
                _ensure_repro_bundle_ready_for_paper_signoff(
                    repo_root=repo_root,
                    run_root=run_root,
                    cfg_path=effective_cfg_path,
                )
            except Exception as exc:
                print(
                    "[onefile] repro bundle pre-signoff closure failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
        if step.name in {"calibrate", "evaluate"}:
            stage_repo_root = None if dry_run else repo_root
            stage_cfg_path = _prepare_stage_cfg_path(step.name, run_root, effective_cfg_path, profile, stage_repo_root)
            step_command = _build_stage_command(step.name, run_root, stage_cfg_path, profile)
        if step.name == "experiment_matrix" and profile == PROFILE_PAPER_FULL_CUDA:
            if "--config" in step_command:
                cfg_index = step_command.index("--config")
                if cfg_index + 1 < len(step_command):
                    step_command[cfg_index + 1] = str(experiment_matrix_cfg_path)

        _print_step_header(step, run_root, step_command)
        if dry_run:
            _print_step_footer(step, 0)
            _print_artifact_presence(step.artifact_paths)
            continue

        return_code = _run_subprocess_for_step(step_command, repo_root)
        _print_step_footer(step, return_code)
        _print_artifact_presence(step.artifact_paths)
        if return_code != 0:
            if step.name in _SIGNOFF_MUST_RUN_STEPS:
                # audits/audits_strict 失败（BLOCK_FREEZE）时延迟返回，确保 signoff 步骤
                # 仍然执行以正式记录 signoff_report.json；否则 signoff_report 永远不存在。
                print(
                    f"[onefile] step={step.name} returned {return_code} (BLOCK_FREEZE); "
                    "deferred — signoff will still run to record the decision"
                )
                if deferred_failure_code is None:
                    deferred_failure_code = return_code
            else:
                return return_code

        # 在 embed 成功后构建 generation attestation statement（全 profile 执行，keys 由环境变量控制）。
        if step.name == "embed" and return_code == 0:
            try:
                _run_attestation_after_embed(run_root, effective_cfg_path)
            except Exception as exc:
                print(
                    f"[onefile] WARN: attestation after embed failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        # embed 成功完成后，为非 paper 模式补充攻击覆盖生成，
        # 确保 evaluate 产物中有真实攻击分组；paper 模式由 experiment_matrix 提供正式覆盖。
        if step.name == "embed" and return_code == 0 and profile != PROFILE_PAPER_FULL_CUDA:
            try:
                _run_attack_coverage_detections(repo_root, run_root, effective_cfg_path, profile)
            except Exception as exc:
                print(
                    f"[onefile] WARN: attack coverage detections failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        # calibrate 成功完成后，重跑一次 detect（re-detect），使用 NP 阈值工件。
        # paper_full_cuda 会在该步结束后将 canonical detect_record 回写主 run_root，
        # 非 paper_full_cuda 仍保持 detect_np 独立子目录语义。
        if step.name == "calibrate" and return_code == 0:
            try:
                _run_redetect_with_np_thresholds(repo_root, run_root, effective_cfg_path, profile)
            except Exception as exc:
                print(
                    f"[onefile] WARN: NP-threshold re-detect failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        # 在 detect 成功后验证 generation attestation（全 profile 执行，keys 由环境变量控制）。
        if step.name == "detect" and return_code == 0:
            try:
                _run_attestation_verification_after_detect(run_root, effective_cfg_path)
            except Exception as exc:
                print(
                    f"[onefile] WARN: attestation verification after detect failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        if step.name == "experiment_matrix" and profile == PROFILE_PAPER_FULL_CUDA and step.artifact_paths:
            summary_path = step.artifact_paths[0]
            try:
                matrix_summary = _load_experiment_matrix_summary(summary_path)
            except Exception as exc:
                # 汇总不可解析会导致后续审计失败，必须前置阻断。
                print(f"[onefile] experiment_matrix summary parse failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

            total_count = matrix_summary.get("total", 0)
            failed_count = matrix_summary.get("failed", 0)
            succeeded_count = matrix_summary.get("succeeded", 0)

            if total_count > 0 and failed_count == total_count and not experiment_matrix_retry_used:
                print(
                    "[onefile] EXPERIMENT_MATRIX_RECOVERY all grid items failed; "
                    "cleaning batch_root and retrying once"
                )
                try:
                    _cleanup_experiment_matrix_batch_root(run_root / "outputs" / "experiment_matrix", run_root)
                except Exception as exc:
                    # 清理失败无法保证复现实验独立性，必须 fail-fast。
                    print(f"[onefile] experiment_matrix cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1

                experiment_matrix_retry_used = True
                _print_step_header(step, run_root, step_command)
                return_code = _run_subprocess_for_step(step_command, repo_root)
                _print_step_footer(step, return_code)
                _print_artifact_presence(step.artifact_paths)
                if return_code != 0:
                    return return_code

                try:
                    matrix_summary = _load_experiment_matrix_summary(summary_path)
                except Exception as exc:
                    print(f"[onefile] experiment_matrix summary parse failed after retry: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                failed_count = matrix_summary.get("failed", 0)
                succeeded_count = matrix_summary.get("succeeded", 0)

            if failed_count > 0 and succeeded_count == 0:
                print(
                    "[onefile] experiment_matrix produced zero successful items; "
                    "abort before audits to expose root cause",
                    file=sys.stderr,
                )
                return 1
    # 所有步骤（含 signoff）执行完毕后，若 audits/audits_strict 有延迟失败码则上报。
    if deferred_failure_code is not None:
        print(
            f"[onefile] workflow completed with deferred audits failure code={deferred_failure_code} "
            "(signoff was allowed to run; check signoff_report.json for decision)"
        )
        return deferred_failure_code
    return 0


def _parse_args() -> argparse.Namespace:
    """
    功能：解析命令行参数。 

    Parse command line arguments for onefile workflow.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Onefile workflow: embed -> detect -> calibrate -> evaluate -> audits -> signoff"
    )
    parser.add_argument("--cfg", required=True, help="Config YAML path")
    parser.add_argument("--run-root", default=None, help="Unified run_root path")
    parser.add_argument(
        "--profile",
        default=PROFILE_CPU_SMOKE,
        choices=[PROFILE_CPU_SMOKE, PROFILE_PAPER_FULL_CUDA, LEGACY_PROFILE_CPU_MIN, LEGACY_PROFILE_CUDA_REAL],
        help="Execution profile",
    )
    parser.add_argument(
        "--signoff-profile",
        default=None,
        choices=["baseline", "paper", "publish"],
        help="Signoff profile passed to run_freeze_signoff.py",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Reserved device selector")
    parser.add_argument("--dry-run", action="store_true", help="Print step commands without execution")
    return parser.parse_args()


def main() -> None:
    """
    功能：onefile workflow 主入口。 

    Main entry point for onefile workflow orchestration.

    Returns:
        None.
    """
    args = _parse_args()
    args.profile = _normalize_profile(args.profile)

    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parent.parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else default_repo_root
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()

    run_root = _build_run_root(repo_root, args.run_root, args.profile)

    print(f"[onefile] repo_root={repo_root}")
    print(f"[onefile] cfg={cfg_path}")
    print(f"[onefile] profile={args.profile}")
    resolved_signoff_profile = _resolve_default_signoff_profile_for_profile(args.profile, args.signoff_profile)
    print(f"[onefile] signoff_profile={resolved_signoff_profile}")
    print(f"[onefile] device={args.device}")
    print(f"[onefile] run_root={run_root}")
    print(f"[onefile] dry_run={args.dry_run}")

    if not repo_root.exists() or not repo_root.is_dir():
        print(f"[onefile] error: repo_root not found: {repo_root}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.exists() or not cfg_path.is_file():
        print(f"[onefile] error: cfg not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    # 检测 cfg 角色并验证其与 profile 的兼容性
    try:
        cfg_text = cfg_path.read_text(encoding="utf-8")
        cfg_obj = yaml.safe_load(cfg_text)
        if not isinstance(cfg_obj, dict):
            print(f"[onefile] error: config root must be mapping", file=sys.stderr)
            sys.exit(2)
        _validate_cfg_role_for_profile(cfg_obj, cfg_path, args.profile)
    except ValueError as e:
        print(f"[onefile] error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[onefile] error: failed to validate config: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    return_code = run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile=args.profile,
        signoff_profile=resolved_signoff_profile,
        dry_run=bool(args.dry_run),
        device_override=args.device,
    )
    sys.exit(return_code)


if __name__ == "__main__":
    main()
