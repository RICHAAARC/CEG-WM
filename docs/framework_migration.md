# Framework Migration Record

## Source

- Upstream tree: <https://github.com/RICHAAARC/RPGov/tree/main/Generative-Watermark>
- Imported from the downloaded `RPGov-main/Generative-Watermark` snapshot on 2026-08-21.
- The downloaded snapshot contains no `.git` metadata, so no source revision is claimed.

The following source digests identify the inspected seed files:

| seed file | SHA-256 |
| --- | --- |
| `governance/policies/dependency_rules.yaml` | `45cd5f037e24372e5b30878846174618eee2fcd937e2711fb359c9fce82f76bf` |
| `governance/policies/notebook_rules.yaml` | `b5fb9a6eb2d813d167e8f5a7550f732502600bdf6f8655a768f05f5276daab79` |
| `governance/harness/audits/audit_dependency_boundaries.py` | `106fcb9d8487f0c6902b8b1ed75e104f9a4bdc3147b40a5c34eb8b1653808024` |
| `governance/harness/audits/audit_notebook_boundaries.py` | `6af99ae2eaf327c626e622256818750554f0b8600c5adcff57101b39e389f804` |
| `governance/tools/extract_release_package.py` | `40895f2b298cf88ced97df989c214d8999207703254f6de652086941f69130fc` |
| `pyproject.toml` | `052b29ac636d991cb87faf2daa7a7978c2730c24681ba26dd036d7947ab3aceb` |

## Retained and rewritten

- policy-driven import boundary checking, adapted to `src/cegwm`;
- notebook location/output/execution-count checks, with naming governance removed;
- real package extraction with sensitive-config and local-path scans;
- extraction tests that import and test a governance-free package;
- lightweight/default versus integration/smoke/slow/formal pytest markers;
- explicit experiment failure records and public key identity only.

## Intentionally omitted

- method-readiness and stage-progression machines;
- mandatory project-skill registry;
- naming, field-registry, placeholder, and root-completeness audits;
- external-baseline comparison governance;
- paper artifact and supported-claim machinery;
- the third extraction profile for paper rebuild;
- all prior CEG-WM method code, thresholds, outputs, and readiness claims.

The target implementation is a semantic rewrite, not a byte-for-byte copy. Seed behavior is retained only where covered by the new repository's tests.
