# Cross-generation-stage evidence protection V2

The 2026-11-10 [memory preparation](stage_protection_v2_memory.md) changes only
component/storage ownership and read-only resource evidence. It does not change
the protocol or authorize retry of STARTED/FAILED work.

Status: local P1 implementation and engineering validation; no real-model P2
result. The user authorized implementation on 2026-11-09 after the four-party
plan audit. The earlier plan-file phrase “audit only” describes its creation
time, not the current implementation authorization. The user subsequently
authorized publication and a notebook that directly executes fixed P2 on Run
all. The source audit chat handles final review/commit/push and the remote link;
this implementation chat does not run real models. Merge and automatic P3/P4
remain outside this delivery.

Authority: `D:/CodexData/research-audits/ceg-full-plan-review-20261109/PLAN_V2.txt`
and `AUDIT_DECISION.txt`. V0/V1 and old research results remain untouched.

## Protocol to implementation

| V2 requirement | Owning implementation |
| --- | --- |
| Two fixed prompts, new generation 2026110911/12, design 2026110920, holdout 2026110921, retained revisions | `configs/stage_protection_v2_colab.json` |
| Two full sources, each branching independently to bit0 and bit1 | `Canary.freeze`, `Canary.trial` in `experiments/run_stage_protection_v2.py` |
| Full20 capture after step8 and step18, scheduler/RNG/conditions | `capture_two` in `src/cegwm/runtime/stage_protection_v2.py` |
| Canonical early zero; historical full RGB diagnostic only | `Canary.freeze`, `historical_vs_canonical` |
| Fixed clean/design target cells chosen once per bit, same old tie rule | `fixed_cells` in `src/cegwm/stage_protection_v2.py` |
| Up to two cumulative early updates; fresh RGB and latent sensitivity after each real candidate | `early_update`, `Canary.trial`, `at_latent` |
| Fresh late zero E, re-read clean/design, fresh sensitivities at E | `Canary.trial`, `StageProtectionProvider.rgb_gradient/latent_gradients` |
| A clean-only; B shared IC/IR; C design half-margin protection | `solve_intervals`, `protection_radius`; one/two-active-face SVD |
| Actual cast, fixed corresponding unmarked denominator, squared budget | `cast_action`, `common_accept` |
| Original proposal and common E fallback; C private IP never rejects B | `Canary.trial`, `common_accept`; original `observation` and `selected` both persisted |
| Holdout only after all proposals/selections frozen | `selection_frozen.pt`, then `Canary.evaluate` |
| Four trial denominator, content-grouped strict C>B and no reverse branch | `grouped_verdict`; active early flip requires different unmarked design blind bit |
| STARTED/attempted/failure/resume and actual nested counts | `stage_protection_store.py`, `physical_counts` |
| User Run all without source upload or unpublished remote dependency | `notebooks/stage_protection_v2_colab.ipynb`, reproducible builder in `scripts/` |
| Cached/env/secret authentication reuse and persistent process output | `src/cegwm/runtime/stage_protection_colab.py` |

The only change to the retained suffix runtime is an optional `after_step`
callback, unused by old paths. It snapshots step18 without another source
forward. No reader, key domain, public preprocessing, old solver or old result
is replaced. There is no dependency on `.codex/` or governance at runtime.

## Exact task and reporting semantics

SD3.5-medium 20 steps, 512 square, guidance 7; SDXL-base strength .3, 20 nominal
steps, guidance 1, empty prompt. Reader: full SD3 VAE mode + unit Rademacher,
key seed 2026110830, QIM delta 1, inner half-width .25, unchanged keyed phase.
Each stage has relative action bound `.01/sqrt(2)` against the corresponding
**full unmarked source** latent norm. Early uses cumulative action from its
original checkpoint. Late uses the modified early branch as its local base,
but the original unmarked step18 norm as denominator. Actual cast is checked
before rendering and checked against the executed action afterward. No strength
repair occurs after a cast violation. Different-time bare L2 norms are not added.

Clean/design fixed-cell membership and global blind bit correctness are separate
fields. `returned_three_channel_success` requires common clean/design IC/IR,
legal budget and correct clean/design/holdout blind bits. Only IC and IR are
defined solver/acceptance cells in V2. For transparency, holdout is also reported
relative to the **same cR** (`fixed_inner`/margin), without choosing a cell from
holdout or adding a new holdout threshold. This precise reporting interpretation
is listed for source-session review. No holdout value can change a proposal,
choice of cell, early stop, fallback or returned image.

`raw_three_channel_success` retains the raw proposal outcome, and returned
success includes explicit common E fallback. A fallback or natural noop cannot
trigger mechanism evidence: the mechanism case additionally needs active early
flip, E needing repair, A repairing clean while leaving design inner, C's actual
action differing from B, nonzero C, accepted C, realized half-margin protection
and holdout bit correctness. Unknown engineering outcomes remain unknown; they
block promotion without being interpreted as scientific infeasibility.

`protection_constraint_active`, C/B requested and actual difference, noop,
infeasible, protection not realized, design margin change, leaving inner and
bit flip are all recorded. Quality reports MSE/PSNR/max RGB change against the
canonical unmarked RGB, with no hard quality gate. `C_minus_B_rgb_mse` exposes
the distortion tradeoff. Even a content-grouped operational signal leaves
`quality_tradeoff_review=REQUIRED`, `P3_authorized=false`, `scientific_pass=null`
and `innovation_pass=null`. No quantitative “obviously worse” quality threshold
is invented; the source audit must assess this unresolved tradeoff.

If an unmarked holdout is unavailable, the overall outcome is
`ENGINEERING_INCOMPLETE`, overall/group operational signals are false, and
quality review is deferred. Trial-only outcome and group signals remain in
explicit diagnostic fields; they cannot constitute overall promotion evidence.

## Run and recover

Local default (no model load):

```bash
PYTHONPATH=src:. python -m experiments.run_stage_protection_v2
```

The notebook's **first cell only mounts Drive**. Its source snapshot is embedded;
no source upload, manual file edits or run checkbox is needed. Default Run all
prepares the environment and directly invokes fixed P2 with `--execute`. The
notebook preselects GPU metadata without choosing/guaranteeing a GPU model or
allocation. It retains Colab CUDA torch and installs the existing pinned model
dependencies. The CLI default above still prints the plan without loading models.

Authentication first reuses environment tokens, the existing Hugging Face login
cache, then Colab `HF_TOKEN`. Lightweight Hub permission checks do not load model
weights. A blocking hidden-token prompt appears only after existing credentials
and anonymous access receive 401/403 for the fixed repositories; network/server
errors are recorded without forcing login. Already-authorized credentials cause
no login prompt. Drive consent, missing model permissions and unavailable GPUs
remain service prerequisites that the notebook cannot grant automatically.
P2 is not run by this implementation chat.

Default output:
`MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01`.
Rerunning reuses exact completed operation artifacts. STARTED/FAILED operations
are not repeated automatically, including after a hard kernel interruption.
Successful downstream-independent work still proceeds; unavailable dependencies
are retained in the original denominator. Config changes cannot mix into an
existing run. One VM must own a run directory: the process lock protects one
Linux VM, and **is not a cross-VM Drive lock guarantee**.

Outputs include `config.json`, `physical_counts.json`, per-operation status and
`result.pt`/`partial.pt`, `selection_frozen.pt/json`, `evaluated.pt`, and
`summary.json`. `runner.log` appends direct child stdout/stderr (including a
traceback on failure), while `setup.log` retains installation output and setup
errors. Prior log invocations are preserved on resume. These files are written
independently of the browser output; shutting down the VM may still stop the run.
Tensor artifacts retain source captures, actual latents/deltas,
RGB images, gradients and diagnostics. They are trusted own-run pickle artifacts,
not intended for loading untrusted uploads. Notebook output displays every
trial/arm, raw proposal images, selections, metrics and failures.

## Physical maxima and unknown resources

| Component | Full four-trial maximum |
| --- | ---: |
| Final E + A/B/C candidates | 16 |
| SD3 forward / denoising steps | 40 / 258 |
| SDXL ordinary / proxy / total forward | 44 / 12 / 56 |
| SDXL actual denoising steps | 336 (6 per pinned real pipeline call) |
| RGB VJP / latent VJP | 16 / 16 |
| Reader result ordinary / auxiliary ordinary / proxy | 70 / 16 / 16 |
| Reader VAE encode total | 102 |

SD3 differentiated forwards are included, not free. RGB and latent VJP are nested
components, not extra independent model calls. SDXL's own input/output VAE
components are separately instrumented and excluded from the reader encode sum.
Attempted is recorded before calls, returned after calls, denoising is counted
at actual transformer/UNet calls. Lower actual counts may reflect failures,
noop/skip or explicit identical-RGB holdout reuse. STARTED operations after abrupt
death remain attempted/incomplete, not successful or automatically retryable.

GPU RAM, host RAM, model-download/storage requirements, real CUDA dtype kernels,
11-step backward feasibility, Colab authentication and elapsed real-model time
are unverified. Large SD3 and SDXL pipelines are released between stages; frozen
reader identity is retained. Saved activations are offloaded to CPU using the
existing runtime, which still consumes host RAM. Old one-step VJP success is not
evidence that the early eleven-step path fits an L4 or any other device.

## Historical failure and next-stage boundary

The retained source05 current-basepoint experiment completed forward and VJP;
both bit0 writes were zero and joint nonzero bit1 still read bit0 on clean,
design and holdout. Finite actual change was far smaller than proxy prediction.
This does not prove real curvature is the cause or all gradients/budgets have
no solution. No old sensitivity is reused at a different RGB basepoint here.

P3's new fixed-margin D, new contents/seeds and P4's zero-bit attribution plus
independent CAL/TEST are conditional future scientific work. This implementation
does not create those executables, run them or infer their results.
