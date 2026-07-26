"""审计项目级 Codex skills 的目录、frontmatter 和必需章节。"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from governance.harness.lib.json_report import build_report, exit_with_report
from governance.harness.lib.project_policy import load_skill_policy


FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<frontmatter>.*?)\n---\n", re.DOTALL)
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def parse_frontmatter(text: str) -> dict[str, str]:
    """解析本项目 skill 使用的简单字符串 frontmatter。"""
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}
    result = {}
    for line in match.group("frontmatter").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"')
    return result


def run_audit(root: str | Path) -> dict:
    root_path = Path(root)
    violations = []
    checked_paths = []

    policy_path = root_path / "governance" / "policies" / "project_skills.yaml"
    checked_paths.append(str(policy_path.relative_to(root_path)))
    try:
        policy = load_skill_policy(root_path)
    except (OSError, ValueError) as error:
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "skill_policy_unreadable", "detail": str(error)})
        return build_report("audit_skill_file_presence", "fail", violations, checked_paths)

    required_policy_fields = ("skill_root", "required_skills", "required_files", "required_sections")
    missing_policy_fields = [field for field in required_policy_fields if field not in policy]
    if missing_policy_fields:
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "skill_policy_field_missing", "fields": missing_policy_fields})
        return build_report("audit_skill_file_presence", "fail", violations, checked_paths)

    skill_root = root_path / policy["skill_root"]
    required_skills = tuple(policy["required_skills"])
    required_files = tuple(policy["required_files"])
    required_sections = tuple(policy["required_sections"])

    if len(required_skills) != len(set(required_skills)):
        violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "duplicate_registered_skill"})

    for relative in policy.get("forbidden_legacy_roots", []):
        checked_paths.append(relative)
        if (root_path / relative).exists():
            violations.append({"path": relative, "reason": "legacy_skill_root_forbidden"})

    actual_skills = {
        path.name
        for path in skill_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    } if skill_root.is_dir() else set()
    if policy.get("unknown_skill") == "fail":
        for skill_name in sorted(actual_skills - set(required_skills)):
            violations.append({"path": str((skill_root / skill_name).relative_to(root_path)), "reason": "unregistered_project_skill"})

    for skill_name in required_skills:
        if not SKILL_NAME_PATTERN.fullmatch(skill_name):
            violations.append({"path": str(policy_path.relative_to(root_path)), "reason": "invalid_registered_skill_name", "skill": skill_name})
            continue
        skill_file = skill_root / skill_name / "SKILL.md"
        metadata_file = skill_root / skill_name / "agents" / "openai.yaml"
        for relative in required_files:
            path = skill_root / skill_name / relative
            checked_paths.append(str(path.relative_to(root_path)))
            if not path.exists():
                violations.append({"path": str(path.relative_to(root_path)), "reason": "required_skill_file_missing"})
        if not skill_file.exists():
            continue

        text = skill_file.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(text)
        if frontmatter.get("name") != skill_name:
            violations.append({"path": str(skill_file.relative_to(root_path)), "reason": "skill_name_mismatch"})
        if not frontmatter.get("description"):
            violations.append({"path": str(skill_file.relative_to(root_path)), "reason": "skill_description_missing"})
        for section in required_sections:
            if section not in text:
                violations.append({"path": str(skill_file.relative_to(root_path)), "reason": "skill_section_missing", "section": section})

        if metadata_file.exists():
            metadata = metadata_file.read_text(encoding="utf-8")
            for field in ("interface:", "display_name:", "short_description:", "default_prompt:"):
                if field not in metadata:
                    violations.append({"path": str(metadata_file.relative_to(root_path)), "reason": "skill_metadata_field_missing", "field": field.rstrip(":")})
            if f"${skill_name}" not in metadata:
                violations.append({"path": str(metadata_file.relative_to(root_path)), "reason": "skill_default_prompt_name_missing"})

    return build_report("audit_skill_file_presence", "fail" if violations else "pass", violations, checked_paths)


def main() -> None:
    exit_with_report(run_audit(Path.cwd()))


if __name__ == "__main__":
    main()
