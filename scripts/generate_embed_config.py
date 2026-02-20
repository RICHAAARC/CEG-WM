"""
生成 Embed 阶段的运行配置文件

功能：根据可配置的参数生成临时运行配置文件（YAML 格式），用于 embed 命令执行。
Module type: Semi-general module
"""

import argparse
import sys
from pathlib import Path
import yaml


def generate_embed_config(
    output_path: str,
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    model_source: str = "hf",
    hf_revision: str = "main",
    inference_prompt: str = "a serene mountain landscape with a lake",
    inference_num_steps: int = 20,
    inference_guidance_scale: float = 4.5,
    seed: int = 42,
    enable_content: bool = True,
    enable_geometry: bool = False,
    enable_paper_faithfulness: bool = True,
    enable_pipeline_inference: bool = True,
    enable_trajectory_tap: bool = True,
) -> dict:
    """
    生成 embed 运行配置文件。

    Generate and save runtime configuration for embed stage.

    Args:
        output_path: Output YAML file path.
        model_id: Hugging Face model ID.
        model_source: Model source ('hf' or 'local').
        hf_revision: Hugging Face revision (branch/tag/commit hash).
        inference_prompt: Text prompt for inference.
        inference_num_steps: Number of diffusion steps (20-28 recommended for SD3).
        inference_guidance_scale: Guidance scale (4.5-7.0 recommended for SD3).
        seed: Random seed for reproducibility.
        enable_content: Enable content chain watermarking.
        enable_geometry: Enable geometry chain watermarking.
        enable_paper_faithfulness: Enable paper faithfulness alignment check.
        enable_pipeline_inference: Enable P1 (pipeline fingerprint).
        enable_trajectory_tap: Enable P2 (trajectory digest).

    Returns:
        Configuration dict.

    Raises:
        ValueError: If parameters are invalid.
        IOError: If output file cannot be written.
    """
    # 参数校验
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be non-empty string")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be non-empty string")
    if model_source not in ("hf", "local"):
        raise ValueError("model_source must be 'hf' or 'local'")
    if not isinstance(hf_revision, str) or not hf_revision:
        raise ValueError("hf_revision must be non-empty string")
    if inference_num_steps < 1 or inference_num_steps > 100:
        raise ValueError("inference_num_steps must be in range [1, 100]")
    if inference_guidance_scale < 0:
        raise ValueError("inference_guidance_scale must be non-negative")
    if not isinstance(inference_prompt, str) or not inference_prompt:
        raise ValueError("inference_prompt must be non-empty string")

    # 当启用 paper_faithfulness 时，必须启用 inference 和 trajectory_tap
    # 因为 paper faithfulness 模式下需要真实的推理和轨迹采样来生成fingerprint和trajectory_digest
    actual_inference_enabled = enable_pipeline_inference
    actual_tap_enabled = enable_trajectory_tap
    if enable_paper_faithfulness:
        actual_inference_enabled = True
        actual_tap_enabled = True

    # 构建配置字典
    runtime_config = {
        # ==============================
        # 核心启用开关
        # ==============================
        "pipeline_build_enabled": True,
        "inference_enabled": actual_inference_enabled,

        # ==============================
        # P2 轨迹采样配置
        # ==============================
        "trajectory_tap": {
            "enabled": actual_tap_enabled,
            "sample_at_steps": [5, 10, 15, 20],
            "sample_layer_names": ["transformer"],
        },

        # ==============================
        # 模型参数（顶层，供 pipeline_factory 读取）
        # ==============================
        "model_id": model_id,
        "model_source": model_source,
        "hf_revision": hf_revision,

        # ==============================
        # 基础配置
        # ==============================
        "policy_path": "content_only",
        "target_fpr": 0.01,

        # ==============================
        # 检测配置
        # ==============================
        "detect": {
            "content": {
                "enabled": enable_content
            },
            "geometry": {
                "enabled": enable_geometry
            }
        },

        # ==============================
        # 水印配置（三链结构）
        # ==============================
        "watermark": {
            "key_id": "master_key_001",
            "pattern_id": "pattern_standard_v1",
            "strength": 0.8,
            "plan_digest": None,

            # ===== LF 链（PRC） =====
            "lf": {
                "enabled": True,
                "codebook_id": "lf_codebook_v1",
                "ecc": "sparse_ldpc",
                "ecc_sparsity": 3,
                "strength": 0.5,
                "variance": 1.5,  # PRC 伪高斯方差（必须 1.5）
            },

            # ===== HF 链（T2SMark） =====
            "hf": {
                "enabled": True,
                "codebook_id": "hf_codebook_v1",
                "ecc": 2,
                "tail_truncation_ratio": 0.1,
                "tail_truncation_mode": "top_k_per_latent",
                "tau": 1.0,
            },

            # ===== 子空间链（Shallow Diffuse） =====
            "subspace": {
                "enabled": True,
                "frame": "latent",
                "selector_id": "selector_dct_v1",
                "grid_rows": 4,
                "grid_cols": 4,
                "grid_h": 64,
                "grid_w": 64,
                "k": 512,
                "topk": 128,
                "score_type": "amplitude",
                "channel_agg": "sum",
                "rank": 8,
                "energy_ratio": 0.9,
                "mask_digest_binding": True,
            },

            # ===== 掩膜配置 =====
            "mask": {
                "threshold": 0.5,
                "resolution_binding": "512x512",
                "postprocess": {
                    "enabled": True,
                    "kernel": "gaussian",
                },
            },

            # ===== 注入点规范 =====
            "injection_site_spec": {
                "hook_type": "callback_on_step_end",
                "target_module_name": "StableDiffusion3Pipeline",
                "target_tensor_name": "latents",
                "hook_timing": "after_scheduler_step",
                "injection_rule_digest": "placeholder",
            },
        },

        # ==============================
        # 推理参数（顶层，供 infer_runtime.py 读取）
        # ==============================
        "inference_prompt": inference_prompt,
        "inference_num_steps": inference_num_steps,
        "inference_guidance_scale": inference_guidance_scale,
        "inference_height": 512,
        "inference_width": 512,

        # ==============================
        # 兼容性字段
        # ==============================
        "generation": {
            "num_inference_steps": inference_num_steps,
            "guidance_scale": inference_guidance_scale,
            "prompt": inference_prompt,
            "seed": seed,
        },
        "model": {
            "height": 512,
            "width": 512,
            "dtype": "float16",
        },
        "enable_xformers": True,
        "evaluate": {
            "decoder_type": "content_correlation",
            "target_fpr": 0.01,
        },
        "impl": {
            "content_extractor_id": "content_baseline_noop_v1",
            "geometry_extractor_id": "geometry_baseline_noop_v1",
            "fusion_rule_id": "fusion_baseline_identity_v1",
            "subspace_planner_id": "subspace_baseline_full_v1",
            "sync_module_id": "geometry_sync_baseline_v1",
        },
    }

    # 如果启用 paper faithfulness，添加标记
    if enable_paper_faithfulness:
        runtime_config["paper_faithfulness"] = {
            "enabled": True,
            "alignment_check": True,
        }

    # 写入 YAML 文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True)

    return runtime_config


def print_config_summary(cfg: dict, config_path: str) -> None:
    """
    打印配置摘要。

    Print configuration summary to stdout.

    Args:
        cfg: Configuration dict.
        config_path: Path to saved config file.
    """
    print("\n" + "=" * 80)
    print("[CONFIG] Embed 运行配置已生成")
    print("=" * 80)

    print(f"\n[文件路径]")
    print(f"  {config_path}")
    print(f"  大小: {Path(config_path).stat().st_size} 字节")

    print(f"\n[模型参数]")
    print(f"  - model_id: {cfg.get('model_id')}")
    print(f"  - model_source: {cfg.get('model_source')}")
    print(f"  - hf_revision: {cfg.get('hf_revision')}")

    print(f"\n[推理参数]")
    print(f"  - inference_prompt: {cfg.get('inference_prompt')}")
    print(f"  - inference_num_steps: {cfg.get('inference_num_steps')}")
    print(f"  - inference_guidance_scale: {cfg.get('inference_guidance_scale')}")

    print(f"\n[Paper Faithfulness 证据配置]")
    pf = cfg.get("paper_faithfulness", {})
    if pf.get("enabled"):
        print(f"  ✅ Paper Faithfulness: 启用")
        inf_en = cfg.get("inference_enabled")
        traj_en = cfg.get("trajectory_tap", {}).get("enabled")
        print(f"    [P1] Pipeline 推理: {'✅ 启用' if inf_en else '❌ 禁用'}")
        print(f"    [P2] 轨迹采样: {'✅ 启用' if traj_en else '❌ 禁用'}")
    else:
        print(f"  ❌ Paper Faithfulness: 禁用")

    print(f"\n[水印链配置]")
    watermark = cfg.get("watermark", {})
    print(f"  - LF（PRC）: {'✅' if watermark.get('lf', {}).get('enabled') else '❌'}")
    print(f"  - HF（T2SMark）: {'✅' if watermark.get('hf', {}).get('enabled') else '❌'}")
    print(f"  - 子空间: {'✅' if watermark.get('subspace', {}).get('enabled') else '❌'}")

    print(f"\n✅ 配置准备完成\n")


def main():
    """CLI 入口点。"""
    parser = argparse.ArgumentParser(
        description="生成 Embed 阶段运行配置文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 YAML 文件路径（必需）",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face 模型 ID",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="hf",
        choices=["hf", "local"],
        help="模型源（hf 或 local）",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="main",
        help="Hugging Face 分支/标签/提交",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a serene mountain landscape with a lake",
        help="推理提示词",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="扩散步数（SD3 推荐 20-28）",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.5,
        help="引导尺度（SD3 推荐 4.5-7.0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--disable-p1",
        action="store_true",
        help="禁用 P1（Pipeline 推理）",
    )
    parser.add_argument(
        "--disable-p2",
        action="store_true",
        help="禁用 P2（轨迹采样）",
    )
    parser.add_argument(
        "--disable-pf",
        action="store_true",
        help="禁用 Paper Faithfulness 对齐检查",
    )

    args = parser.parse_args()

    try:
        cfg = generate_embed_config(
            output_path=args.output,
            model_id=args.model_id,
            model_source=args.model_source,
            hf_revision=args.hf_revision,
            inference_prompt=args.prompt,
            inference_num_steps=args.steps,
            inference_guidance_scale=args.guidance,
            seed=args.seed,
            enable_pipeline_inference=not args.disable_p1,
            enable_trajectory_tap=not args.disable_p2,
            enable_paper_faithfulness=not args.disable_pf,
        )
        print_config_summary(cfg, args.output)
        return 0
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
