GDS v2.1 — G0 engineering gate (2026-11-28)

Authority: PROPOSAL_V2.txt plus AMENDMENTS_V2_1.txt in the 2026-11-28
three-context review. This implementation covers G0 only.

Local:
  python g0_cpu.py --output NEW_EMPTY_OUTPUT_DIRECTORY

CUDA/Colab (only after the CPU report passes for these exact files):
  python g0_gpu.py --cpu-report CPU_OUTPUT/report.json --output NEW_GPU_DIRECTORY

The GPU gate makes exactly TWO native generations of ONE diagnostic prompt in
the same process/GPU/runtime: baseline vs actual-kernel observer. No watermark is
written. It compares all 100 token and RNG states, the decoded image, and replays
the native Gumbel draw from the captured raw kernel with a CLONED RNG. The frozen
t=60..11 window is inspected for actual eligible input/output pools. This is not
RGB readability, robustness, quality equivalence, or research novelty evidence.

CPU checks use the real scheduler implementation with synthetic small logits,
plus exact small-state probability enumeration and full-grid geometry. These
checks do NOT substitute for real checkpoint/GPU verification. A failure, missing
CUDA, mismatch or absence of observed native exchange opportunities blocks G1.
No automatic rerun or protocol relaxation is implemented. Failed attempts remain
in report.json; logs and traces are retained. Existing outputs are never resumed.

The two G0 diagnostic generations are distinct from the separately authorized G1
ceiling: 8 NEW sources x N/B0/B1/B2/P/P-E = 48 generation attempts. This delivery
contains no G1 runner and no reader/training/TrustMark code. G1 notebook preparation
awaits the actual G0 report. Do not reuse the G0 diagnostic source in G1.

Use current Colab GPU runtime; do not restore the old CEG training environment or
replace Colab's torch/CUDA installation. Pin diffusers 0.39.0 / transformers 4.57.6.
Legacy VQ component naming and norm_type are handled in a temporary config view;
original Hub cache and weights are unchanged. Existing discrete-pair model cache
may be reused. This loader contains no remote-code execution flag.

Files with private_trace in their names are engineering artifacts; they are never
blind-detector inputs. Reports explicitly distinguish local, GPU and future G1.
