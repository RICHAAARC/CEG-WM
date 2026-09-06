#!/usr/bin/env bash
# Invoke only after separate authorization for real GPU preflight.
set -euo pipefail
if [[ "${1:-}" != "--execute-authorized-preflight" ]]; then
  echo "Required: --execute-authorized-preflight" >&2
  exit 2
fi
: "${CEG_WM_ROOT_KEY:?Set CEG_WM_ROOT_KEY without printing it}"
: "${HF_TOKEN:?Set HF_TOKEN without printing it}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${ROTATION_PREFLIGHT_RUNTIME_ROOT:-/content/rotation-diagnostic-runtime}"
output="${ROTATION_PREFLIGHT_OUTPUT:-/content/drive/MyDrive/CEG-WM/RotationFailure-Diagnostic-V1/preflight-v1}"
if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Mount Drive before preflight." >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Existing output retained; audit it before authorizing another attempt." >&2
  exit 2
fi
# Hard wall-clock cap includes model loading. No automatic retry.
timeout --signal=TERM --kill-after=30s 30m python "$script_dir/preflight.py" \
  --runtime-root "$runtime_root" --output "$output" --execute-authorized-preflight
