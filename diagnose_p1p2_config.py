"""
诊断 P1/P2 配置传递问题

Module type: General module

功能说明：
- 检查 YAML 配置文件中的 inference_enabled 和 trajectory_tap 字段是否被正确定义
- 验证这些字段是否在代码中被正确读取
- 诊断配置生成和传递链路
"""

import json
import yaml
from pathlib import Path

def diagnose_p1p2_config():
    """
    诊断 P1/P2 配置链路：
    1. 检查 embed_record.json 中的配置信息
    2. 匹配实际的 run_meta.cfg 内容
    3. 输出诊断报告
    """
    print("=" * 80)
    print("诊断 P1/P2 配置问题")
    print("=" * 80)
    
    # 定位 embed_record.json - 检查多个可能的位置
    audit_dir_options = [
        Path("audit_evidence"),  # 最新运行（直接在 audit_evidence 下）
        Path("audit_evidence/2026-02-20-2000"),  # Colab 格式（日期子目录）
    ]
    
    audit_dir = None
    for option in audit_dir_options:
        if option.exists():
            audit_dir = option
            break
    
    if not audit_dir:
        print(f"\n❌ 审计目录不存在")
        print(f"   尝试了: {[str(o) for o in audit_dir_options]}")
        return
    
    print(f"\n✅ 找到审计目录: {audit_dir}")
    
    # 查找 embed_record.json
    records_dir = audit_dir / "run_root" / "records"
    embed_record_files = list(records_dir.glob("embed_*.json"))
    
    if not embed_record_files:
        print(f"\n❌ 未找到 embed_record.json 文件")
        return
    
    embed_record_path = embed_record_files[0]
    print(f"\n✅ 找到 embed_record: {embed_record_path.name}")
    
    with open(embed_record_path, "r", encoding="utf-8") as f:
        embed_record = json.load(f)
    
    # 提取关键配置信息
    print("\n" + "=" * 80)
    print("【配置段 1】Run Metadata 中的配置")
    print("=" * 80)
    
    cfg = embed_record.get("run_meta", {}).get("cfg", {})
    
    if not cfg:
        print("\n⚠️  未找到 run_meta.cfg 字段")
        return
    
    # 检查 pipeline 相关配置
    print(f"\n[Pipeline 构建配置]")
    pipeline_build_enabled = cfg.get("pipeline_build_enabled")
    print(f"  - pipeline_build_enabled: {pipeline_build_enabled}")
    if pipeline_build_enabled is None:
        print(f"    ⚠️  缺失！应该为 True（默认值）")
    elif not pipeline_build_enabled:
        print(f"    ❌ 被设置为 False，管道不会被构建！")
    else:
        print(f"    ✅ 启用（True）")
    
    # 检查推理配置
    print(f"\n[推理配置]")
    inference_enabled = cfg.get("inference_enabled")
    print(f"  - inference_enabled: {inference_enabled}")
    if inference_enabled is None:
        print(f"    ⚠️  缺失！默认为 False，需要显式设置为 True")
    elif not inference_enabled:
        print(f"    ❌ 被设置为 False，推理被禁用！这是 P1/P2 NA 的根本原因")
    else:
        print(f"    ✅ 启用（True）")
    
    # 检查轨迹采样配置
    print(f"\n[轨迹采样配置]")
    trajectory_tap = cfg.get("trajectory_tap", {})
    if not trajectory_tap:
        print(f"  - trajectory_tap: <未定义>")
        print(f"    ⚠️  缺失或为空，P2 会是 NA")
    else:
        tap_enabled = trajectory_tap.get("enabled")
        print(f"  - trajectory_tap.enabled: {tap_enabled}")
        if not tap_enabled:
            print(f"    ❌ 被设置为 False，轨迹采样被禁用！")
        else:
            print(f"    ✅ 启用（True）")
            
        sample_steps = trajectory_tap.get("sample_at_steps")
        sample_layers = trajectory_tap.get("sample_layer_names")
        print(f"  - trajectory_tap.sample_at_steps: {sample_steps}")
        print(f"  - trajectory_tap.sample_layer_names: {sample_layers}")
    
    # 检查生成配置
    print(f"\n[推理参数配置]")
    inference_prompt = cfg.get("inference_prompt") or cfg.get("generation", {}).get("prompt")
    inference_num_steps = cfg.get("inference_num_steps") or cfg.get("generation", {}).get("num_inference_steps")
    inference_guidance = cfg.get("inference_guidance_scale") or cfg.get("generation", {}).get("guidance_scale")
    
    print(f"  - inference_prompt: {repr(inference_prompt)[:50]}..." if inference_prompt else "  - inference_prompt: <缺失>")
    print(f"  - inference_num_steps: {inference_num_steps}")
    print(f"  - inference_guidance_scale: {inference_guidance}")
    
    if not all([inference_prompt, inference_num_steps, inference_guidance]):
        print(f"\n  ⚠️  某些推理参数缺失或不完整")
    
    # 提取 pipeline 和推理结果
    print("\n" + "=" * 80)
    print("【执行结果 1】Pipeline 构建结果")
    print("=" * 80)
    
    run_meta = embed_record.get("run_meta", {})
    pipeline_status = run_meta.get("pipeline_status")
    pipeline_error = run_meta.get("pipeline_error")
    
    print(f"\n  - pipeline_status: {pipeline_status}")
    print(f"  - pipeline_error: {pipeline_error}")
    
    if pipeline_status == "unbuilt":
        print(f"\n  ❌ Pipeline 未被构建！")
        if not inference_enabled:
            print(f"     原因：inference_enabled=false（或未定义）")
    elif pipeline_status == "failed":
        print(f"\n  ❌ Pipeline 构建失败！")
        print(f"     错误：{pipeline_error}")
    elif pipeline_status == "built":
        print(f"\n  ✅ Pipeline 构建成功")
    
    # 提取推理结果
    print("\n" + "=" * 80)
    print("【执行结果 2】推理执行结果")
    print("=" * 80)
    
    content_ev = embed_record.get("content_evidence", {})
    infer_trace = content_ev.get("infer_trace", {})
    
    inference_status = infer_trace.get("inference_status")
    inference_error = infer_trace.get("inference_error")
    
    print(f"\n  - inference_status: {inference_status}")
    print(f"  - inference_error: {inference_error}")
    
    if inference_status == "disabled":
        print(f"\n  ❌ 推理被禁用！")
        print(f"     这就是为什么 P1/P2 都是 NA")
        if not inference_enabled:
            print(f"     根本原因：inference_enabled 在配置中为 false（或缺失默认值）")
    elif inference_status == "failed":
        print(f"\n  ❌ 推理失败！")
        print(f"     错误：{inference_error}")
    elif inference_status == "ok":
        print(f"\n  ✅ 推理执行成功")
    
    # 检查证据状态
    print("\n" + "=" * 80)
    print("【证据状态】P1/P2 关键证据")
    print("=" * 80)
    
    pfp = content_ev.get("pipeline_fingerprint", {})
    pfp_status = pfp.get("status")
    pfp_reason = pfp.get("reason")
    
    print(f"\n[Pipeline Fingerprint]")
    print(f"  - status: {pfp_status}")
    print(f"  - reason: {pfp_reason}")
    
    traj_ev = content_ev.get("trajectory_evidence", {})
    traj_status = traj_ev.get("status")
    traj_reason = traj_ev.get("trajectory_absent_reason")
    
    print(f"\n[Trajectory Evidence]")
    print(f"  - status: {traj_status}")
    print(f"  - absence_reason: {traj_reason}")
    
    # 诊断总结
    print("\n" + "=" * 80)
    print("【诊断总结】")
    print("=" * 80)
    
    if inference_status == "disabled":
        print(f"\n🔴 根本原因：inference_enabled=false（或缺失）")
        print(f"\n解决方案：")
        print(f"  1. 在 Notebook Cell E 中，确保配置包含：")
        print(f"     runtime_config = {{")
        print(f"       ...")
        print(f"       \"inference_enabled\": True,  # ← 添加此行")
        print(f"       \"trajectory_tap\": {{")
        print(f"         \"enabled\": True,          # ← 添加此行")
        print(f"         \"sample_at_steps\": [5, 10, 15, 20],")
        print(f"       }},")
        print(f"       ...}}")
        print(f"\n  2. 重新运行 Cell E 生成配置")
        print(f"  3. 重新运行 Cell F 执行 embed")
        print(f"\n  4. 如果仍然失败，检查：")
        print(f"     - embed.log 是否包含 'inference_enabled' 的日志")
        print(f"     - 配置文件 (temp_runtime.yaml) 的内容是否正确")
        print(f"     - run_embed 命令是否正确传递了 --config 参数")
    elif inference_status == "failed":
        print(f"\n🔴 根本原因：推理执行失败")
        print(f"失败信息：{inference_error}")
        print(f"\n解决方案：")
        print(f"  1. 检查 embed.log 最后 100 行的错误信息")
        print(f"  2. 常见原因：")
        print(f"     - 模型下载失败或路径不正确")
        print(f"     - GPU 内存不足")
        print(f"     - 推理参数设置不当（如 num_steps 过大）")
        print(f"  3. 使用以下命令查看详细日志：")
        print(f"     tail -100 audit_evidence/2026-02-20-2000/run_logs/embed.log | grep -i 'error\\|failed\\|exception'")
    elif inference_status == "ok":
        print(f"\n✅ 推理已成功执行")
        print(f"但 P1/P2 仍为 NA？")
        print(f"  这可能说明 pipeline 构建被跳过了")
        print(f"  检查 pipeline_status: {pipeline_status}")
    else:
        print(f"\n❓ 未知状态：{inference_status}")

if __name__ == "__main__":
    diagnose_p1p2_config()
