"""
功能：攻击协议可实现性审计（静态阻断项）

Module type: Core innovation module

校验 configs/attack_protocol.yaml 声明的所有攻击族与参数版本是否可被 attack_runner 实现。
任何协议—实现不一致必须 FAIL 并阻断发布。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# 添加 repo 到 sys.path 以支持 main 包导入
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.evaluation import protocol_loader
from main.evaluation import attack_coverage
from main.evaluation import attack_protocol_guard


AUDIT_ID = "attack.protocol_implementable"
GATE_NAME = "attack.protocol_implementable"
SEVERITY = "BLOCK"
CATEGORY = "B"


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """
    功能：执行攻击协议可实现性审计。

    Run attack protocol implementability audit.

    Args:
        repo_root: Repository root directory.

    Returns:
        Structured audit result.
    """
    try:
        # (1) 加载协议与覆盖声明
        attack_protocol_path = repo_root / "configs" / "attack_protocol.yaml"
        if not attack_protocol_path.exists():
            return {
                "audit_id": AUDIT_ID,
                "gate_name": GATE_NAME,
                "category": CATEGORY,
                "severity": SEVERITY,
                "result": "FAIL",
                "rule": "attack protocol fact source must exist",
                "evidence": {
                    "expected_path": str(attack_protocol_path),
                    "exists": False,
                },
                "impact": "cannot validate protocol-implementation consistency",
                "fix": "create configs/attack_protocol.yaml with valid protocol spec",
            }

        # 使用临时 cfg 加载协议（仅用于静态审计）
        temp_cfg = {
            "evaluate": {
                "attack_protocol_path": str(attack_protocol_path),
            }
        }
        protocol_spec = protocol_loader.load_attack_protocol_spec(temp_cfg)
        coverage_manifest = attack_coverage.compute_attack_coverage_manifest()

        # (2) 执行一致性校验
        attack_protocol_guard.assert_attack_protocol_is_implementable(
            protocol_spec,
            coverage_manifest,
        )

        # (3) 校验通过
        return {
            "audit_id": AUDIT_ID,
            "gate_name": GATE_NAME,
            "category": CATEGORY,
            "severity": "BLOCK",
            "result": "PASS",
            "rule": "attack protocol declares only implementable families and params_versions",
            "evidence": {
                "protocol_version": protocol_spec.get("version", "<absent>"),
                "protocol_digest": protocol_spec.get("attack_protocol_digest", "<absent>"),
                "coverage_digest": coverage_manifest.get("attack_coverage_digest", "<absent>"),
                "supported_families": sorted(coverage_manifest.get("supported_families", [])),
                "protocol_families": sorted(protocol_spec.get("families", {}).keys()),
            },
            "impact": "N.A.",
            "fix": "N.A.",
        }

    except RuntimeError as exc:
        # 协议—实现不一致，必须 FAIL
        error_message = str(exc)
        evidence = {}
        if "Evidence:" in error_message:
            evidence_str = error_message.split("Evidence:")[-1].strip()
            try:
                import ast
                evidence = ast.literal_eval(evidence_str)
            except Exception:
                evidence = {"error_message": error_message}
        else:
            evidence = {"error_message": error_message}

        return {
            "audit_id": AUDIT_ID,
            "gate_name": GATE_NAME,
            "category": CATEGORY,
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "attack protocol declares unimplementable attacks",
            "evidence": evidence,
            "impact": "protocol-implementation inconsistency would cause evaluation failures",
            "fix": "remove unsupported families from configs/attack_protocol.yaml or implement in attack_runner",
        }

    except Exception as exc:
        # 未预期异常，必须 FAIL（审计脚本本身故障）
        return {
            "audit_id": AUDIT_ID,
            "gate_name": GATE_NAME,
            "category": CATEGORY,
            "severity": SEVERITY,
            "result": "FAIL",
            "rule": "audit script execution exception",
            "evidence": {
                "error": f"{type(exc).__name__}: {exc}",
            },
            "impact": "cannot validate protocol-implementation consistency",
            "fix": "fix audit script or protocol loader",
        }


def main() -> None:
    """CLI entry for attack protocol implementability audit."""
    if len(sys.argv) < 2:
        print("Usage: python audit_attack_protocol_implementable.py <repo_root>", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    result = run_audit(repo_root)

    # 输出 JSON 结果
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 根据结果设置退出码
    if result.get("result") == "PASS":
        sys.exit(0)
    elif result.get("result") == "N.A.":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
