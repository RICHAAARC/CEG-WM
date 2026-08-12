# Experiment execution servers

## Disabled-routing content-combination directional diagnosis

`content_uniform_combination_directional_diagnosis_server.py` 是当前 Colab 与普通 GPU
服务器共用的执行入口；当前唯一授权 Notebook 为
`notebooks/colab/content_uniform_combination_directional_diagnosis.ipynb`，固定绑定
execution revision `b242261f10a034b541c571afd30b91b77eaddf19` 与全新 run ID
`ceg_wm_content_uniform_combination_directional_diagnosis`。服务器执行 1 个 operational、
32 个 clean cross-fit reference 与 8 个 six-image scientific probes，共 41 units，每 unit
仅 attempt zero；Notebook 仅负责 Drive、Secrets、exact checkout、调用及 create-only
ZIP/receipt/`SHA256SUMS` 导出，失败也先导出再报错。

每个 probe 分别冻结 `a = 0.25, 0.50, 0.75`，记录 C0、C1(w) 和 C2；其中 `a` 是
embed coefficient，`w` 仅是 C1 detector weight。本门不选择 `a`、`w` 或组合函数，
正式 detector 仍为 HF-only；通过仅允许另行申请 candidate selection，不拟合 formal
threshold/FPR，也不形成 promotion、calibration、evaluation、baseline、joint 或论文 claim。

## Content-routing directional diagnosis

`content_routing_directional_diagnosis_server.py` 保留已完成的独立 routing diagnosis，
当前为 **paused / not authorized**。它此前是 Colab 与普通 GPU 服务器共用的
执行入口；此前唯一授权 Notebook 为
`notebooks/colab/content_routing_directional_diagnosis.ipynb`，固定绑定 execution revision
`925c2cbc727e3b18e91c0b3981eeed1b470a955a` 与全新独立 run ID
`ceg_wm_content_routing_positive_reference_support_correction_diagnosis`。服务器执行冻结的 2 个 operational、32 个
reference-fit 与 8 个 paired routed/uniform scientific units，共 42 units、每 unit 仅
attempt zero。它负责依赖、模型、真实 public method/runtime、records、create-only persistence、
package 与 result/diagnostic ZIP；Notebook 只负责 Drive、Secrets、exact checkout、调用和
ZIP/receipt/`SHA256SUMS` 导出，失败也必须先保存 diagnostic export。

原始 T/R/Q records 保留 finite nonnegative 值（包括零）。每个 probe fold 将 fold 外 24 条
records 的完整空间值展平后，仅 strictly positive 子集进入一次 exact nearest-rank P95，最终
P95 必须大于零；缺少正值支持属于 implementation/dependency blocked，不是科学阴性。

全部 42 units 与 8 个 paired probe 的 routed/uniform 两条 arm 都固定使用 `a = 0.50`；
这是本次 routing 因果控制，不是 alpha selection、组合权重选择或跨 alpha 外推。通过时只
允许另行申请 fixed-half routing directional validation。该 development diagnosis 不拟合
threshold 或 FPR，不形成正式组合、candidate promotion、calibration、evaluation、baseline
或论文 claim。旧 run `ceg_wm_content_routing_backend_binding_correction_diagnosis`、artifacts 与 records
保持 producer-bound；当前入口不读取、恢复、迁移、改写或拼接进本次全新 run 的固定分母。

## Q/K synchronization-write diagnosis

`qk_synchronization_write_diagnostic_server.py` 保留此前 Colab 与普通 GPU 服务器共用的
执行入口，但其 Notebook 当前为 **paused / not authorized**。此前授权 Notebook 为
`notebooks/colab/qk_synchronization_write_diagnostic.ipynb`，固定绑定 execution
revision `24042298bef550803c1710b84485c07ca6223cf2` 与独立 run ID
`ceg_wm_qk_vae_checkpoint_operation_localization`。当前入口仅用于验证已审核的
suffix-memory correction，服务器的 execution authority 精确为
`1 operational / 0 scientific / 1 total / 1 attempt`：只运行 unit0 attempt0，并在
operational success 或安全 failure diagnostic 后立即停止；休眠的 12 个 ratio 与16个
transform units仍保留在科学protocol定义中，但本入口不注册、不执行，也不产生aggregate。
旧 run ID `ceg_wm_qk_runtime_failure_localization`、
`ceg_wm_qk_synchronization_write_public_rgb8_diagnosis` 与更早的
`ceg_wm_qk_synchronization_write_diagnosis` 下的 records、diagnostics 与 intents
保持不可变；当前服务器入口不读取、迁移、覆盖或混入这些历史执行内容。

服务器负责基本 GPU/磁盘检查、冻结依赖安装、配置模型 revision 下载、真实 public
method/runtime 调用、operational record、持久化和内部 result 或 diagnostic ZIP。
Notebook 只负责 Drive、Secrets、exact checkout、调用与 export receipt/ZIP/`SHA256SUMS`。
本 memory-correction preflight 不形成Q/K机制阳性或阴性，不让几何产生水印阳性，不形成 transform estimator
结论、threshold、FPR、candidate promotion、calibration、formal evaluation、baseline
或论文 claim。

## Completed LF whitened directional validation

`lf_whitened_directional_validation_server.py` 与
`notebooks/colab/lf_whitened_directional_validation.ipynb`，固定绑定 execution revision
`51adb765cdddafcb4c65c357e899c77b4c9f36d2` 与优化后独立 run ID
`ceg_wm_lf_whitened_directional_validation_prepared_feature_execution`，已完成冻结的 1 个 non-scientific
public-endpoint smoke 与 32 个 LF whitened directional scientific units；每 unit 最多
2 attempts、2700 秒。该入口当前为 **paused / not authorized**；operational unit 不计
scientific coverage，历史 records 不读取、不迁移、不改写或混入当前 content-combination 诊断分母。

服务器负责基本 GPU/磁盘检查、冻结依赖安装、配置模型 revision 下载、真实 public
method/runtime 调用、正式 records、跨 session persistence 和内部 result 或 diagnostic ZIP。
Notebook 只负责 Drive、Secrets、exact checkout、调用与 export receipt/ZIP/`SHA256SUMS`。
本 development-only directional validation 只读使用 producer revision
`a78c47184cf83ad351bb4442ebd31c218726de25`、run ID
`ceg_wm_lf_whitening_asset_fit_and_score_screening` 已冻结的 whitening asset，不重拟合、
迁移或回写旧 run。它不拟合 threshold，不授权 FPR、candidate promotion、calibration、
formal evaluation、baseline 或论文 claim；不执行 routing、LF/HF 组合、Q/K、estimator、
reliability、rectification 或 conditional recovery。除当前 content-combination Notebook 外，其他
checked-in Notebook 入口均已暂停。
旧 execution revision `194eccdd1f16c295528a4d9e1d7c75c2748f061a` 与旧 run ID
`ceg_wm_lf_whitened_directional_validation` 保持 producer-bound 历史身份，当前为
**paused / not authorized**，不得续跑或与优化后 run 混合。

`lf_whitened_score_screening_server.py` 与
`notebooks/colab/lf_whitened_score_screening.ipynb` 已完成 1 个 non-scientific operational
smoke、32 个 clean null-fit 与 8 个 paired raw-vs-whitened screening units，当前为
**paused / not authorized**。其冻结 whitening asset 仅作为当前方向验证的只读输入；旧
records 保持原 producer/run 身份，且不进入新验证的 scientific 分母。

`lf_transmission_diagnostic_server.py` 与
`notebooks/colab/lf_transmission_diagnostic.ipynb` 绑定的 execution revision
`2337f9d7c773a6054d558108e31d07d35fbee42f`、run ID
`ceg_wm_lf_carrier_to_detector_transmission_diagnostic` 已完成历史诊断职责，当前为
**paused / not authorized**。其 records 保持独立，不读取、不迁移、不改写或混入当前
LF whitening fit 与 screening 分母。

`hf_only_detector_directional_validation_server.py` 与
`notebooks/colab/hf_only_detector_directional_validation.ipynb` 绑定的 execution revision
`0d4253ab2614c642563c566e6268565c337b503f`、run ID
`ceg_wm_hf_only_detector_directional_validation_binary32_budget_authority` 当前为
**paused / not authorized**。更早的 run
`ceg_wm_hf_only_detector_directional_validation_initial_gate` 及其 records 是
immutable partial evidence；当前 LF 入口不读取、不迁移、不改写或混合这些 namespace。

`hf_transmission_diagnostic_server.py` 与 `notebooks/colab/hf_transmission_diagnostic.ipynb`
保留其已完成的历史传输诊断身份，
当前 **paused / not authorized**，不得续跑或与 directional validation records 混合。
历史 execution revision 为 `af1eea8f55086b583e3e5e4a02586959983db70b`，run ID 为
`ceg_wm_hf_transmission_diagnostic_server_execution`。

## Development exploration

`development_exploration_server.py` is the Colab-neutral and server-direct entrypoint for
the frozen 13-module development exploration. Given one clean exact repository revision,
an absolute persistent root, an ephemeral cache root, a fixed run ID, and a unique session
ID, it checks basic GPU and disk availability, installs the version-frozen dependency list
without hash mode, downloads the configured model ID and revision without model-file hash
validation, then calls `development_exploration_entrypoint.py`.

The entrypoint owns the formal runner, records, create-only intent/bundle/`COMMITTED`
protocol, cross-session recovery, and result or diagnostic ZIP. It observes the frozen
21-hour soft stop and 24-hour hard cap without changing the unit or attempt budget.
`HF_TOKEN` and `CEG_WM_ROOT_KEY` are passed only through the worker environment and never
persisted in receipts or artifacts. Colab-side copying of ZIP, receipt, and `SHA256SUMS`
is a delivery convenience; only verified persistent `COMMITTED` bundles establish unit
completion.

The checked-in `development_exploration.ipynb` operational Notebook completed its authorized
boundary and is now permanently paused and not authorized for another **Run all**. It invoked
the server from detached execution revision
`7e449aa29f53ea38e3a044681c75c8f3dccff135` with run ID
`ceg_wm_thirteen_module_mechanism_screening_session_resume_validation` and always passes
both `--maximum-wiring-clusters 2` and `--stop-before-scientific-units`. The first session
is therefore limited to units 0 through 3, two preflight and two wiring units: four operational units and zero
scientific units. A later Run all in the same new namespace validates immediate recovery from
the preceding verified session receipt. Later sessions resume at most two wiring units each until all ten operational
screening units are committed. Every session stops before scientific unit 10, and repeated
Run all after operational completion was not used to create a scientific artifact. The verified
result contains two preflight units and all 8/8 wiring smoke clusters and terminates with
`authorized_operational_boundary_reached`. These units receive no module-science credit.

The checked-in `thirteen_module_mechanism_screening.ipynb` completed its historical full-screening
run and is now paused and not authorized to run again. It invoked this same
server and exact execution revision with the fresh run ID
`ceg_wm_thirteen_module_mechanism_scientific_screening`, begins at frozen roster unit 0, and passes
neither `--maximum-wiring-clusters` nor `--stop-before-scientific-units`. The server therefore owns
the complete deterministic roster, session recovery, soft stop, records, and budget. The new run
does not read, resume, migrate, rewrite, delete, or mix the completed operational namespace or any
historical run. The frozen study budget is 240 scientific
plus 42 operational units, 282 total, with 846 maximum attempts. The prior 506-unit
development authority is historical and is neither the active entrypoint nor the current
budget denominator. The prior
execution revision `2ff836f45c4012010092f7075e749507ae2ad9ae`, run
`ceg_wm_thirteen_module_mechanism_screening`, and its dangling intent are immutable
diagnostics. The new operational-validation run does not read, resume, migrate, rewrite, or
delete them. The prior
unexecuted delivery revisions `ce536f1ad66b5f45c05d7b0a08e5c83fb8fb4b29` and
`6c84cb121030a1190a183955dd4a27798a0eb975`, together with recovery namespace
`ceg_wm_thirteen_module_mechanism_screening_preflight_recovery`, also remain unchanged and
unapproved for execution. The prior `b66cb04ebb41f0d5473c498ad5769b467ff26d7e` run
`ceg_wm_thirteen_module_mechanism_screening_operational_validation`, including its four
committed operational units and second-session active-writer diagnostic, also remains immutable.
The new session-resume-validation run does not read, resume, migrate, rewrite, delete, or mix
any of those namespaces. The prior
`ceg_wm_development_exploration_detector_crossfit_execution` run and all of its
scientific records, operational records, and diagnostic artifacts remain unchanged and
are not read, migrated, rewritten, or deleted. The prior
`ceg_wm_development_exploration_science_first_v42` run namespace, records, dangling
attempts, and full artifacts remain unchanged and are not read or migrated. The prior
`ceg_wm_development_exploration_scientific_execution` run remains unchanged with two
operational commits, zero scientific commits, dangling unit 0002 attempt 0, and diagnostic
`builtins.AssertionError`. Any existing
`ceg_wm_development_exploration_joint_record_execution` directory also remains unchanged;
neither prior run is read, migrated, or deleted.
That execution authority is intentionally
separate from the later Notebook delivery revision and must not be replaced by a mutable
branch.

Rebuilding the deterministic tracked-tree execution package from that exact revision
produces 4,549,335 bytes with SHA-256
`260a76d0e10ddbcf705bbdfda11e5593c688d2b3957d1635b4404b498187067e`.

## HF-only threshold-fit GPU execution

The separate schema-v2 HF-only threshold-fit GPU execution delivery
package. It executes one preregistered fit shard at an exact committed
revision. It cannot approve tau, unlock untouched-confirmation data, or support
a scientific claim by itself. Its checked-in `experiment_execution.ipynb` remains an
authoritative historical entrypoint but is paused and not authorized to run in the current
mechanism-screening batch. The checked-in `runtime_qualification.ipynb` likewise preserves
the audited runtime-qualification authority but is paused and not authorized to rerun.

## Frozen dependency closure

`requirements_hf_only_threshold_fit_gpu_execution.txt` is the hf_only_reference_validation-specific complete transitive
dependency lock for Linux x86_64, CPython 3.12, and the frozen SD3.5 Colab GPU
candidate. It contains 62 exact distributions, including
`torch==2.11.0+cu128`, and has SHA-256
`07a4c1bbe6fc5e7e6b38334c5a9919a8565b810a9aae7820b61c24cee91270de`.

The lock was generated from the eight frozen top-level requirements with pip's
resolver in dry-run report mode. Resolution used PyPI, the official PyTorch
cu128 index, and the official NVIDIA index, with CPython 3.12 ABI and explicit
Linux wheel tags `manylinux_2_28_x86_64`, `manylinux_2_27_x86_64`,
`manylinux_2_18_x86_64`, and `manylinux2014_x86_64`. The lock entries are the
normalized, sorted name/version pairs from all 62 `install` records in that
resolver report, not only the eight requested distributions.

Before package import, the package-external bootstrap verifies the lock digest
and exact syntax. It reuses the global environment only when every lock entry
has the exact version. Otherwise it installs every locked distribution into
ephemeral storage with `--no-deps`, then requires exact equality of the target
distribution set and all versions. The entrypoint repeats the frozen-lock and
installed-version checks and records all 62 versions plus the exact imported
torch local version in the execution facts.

## Build and execute

`build_experiment_execution_package.py` reads only tracked blobs from one clean
exact commit and writes a deterministic external ZIP plus adjacent delivery
sidecar. Its exact allowlist includes the hf_only_reference_validation lock and excludes the shared
runtime-qualification lock, untouched-confirmation manifest, baselines,
comparison protocol, synthetic runtime, governance, Notebook, checked-in
outputs, builder, and package-external bootstrap.

After independent review, build to a new path outside the repository:

```bash
python scripts/experiment_execution/build_experiment_execution_package.py \
  --root . \
  --output-zip '<outside-repository>/ceg_wm_hf_only_threshold_fit.zip' \
  --committed-revision '<exact 40-hex HEAD>'
```

Independently record the SHA-256 of the external bootstrap, archive, sidecar,
and embedded manifest before upload. The thin output-free Colab Notebook passes
those trust values, the exact revision, run ID, shard index, and Secrets to the
separately distributed schema-v2 bootstrap. It does not install dependencies,
unpack the archive, import the package, or validate its own result.

Each invocation runs one frozen shard. Persistent attempt records bind exact
revision, run, and shard identities for resume; every result or diagnostic ZIP
uses a unique name. A bootstrap, resource, execution, exclusion, incomplete, or
scientific diagnostic is not a successful shard and cannot be promoted into
method evidence.

Historical non-hf_only_reference_validation runtime-qualification materials remain in their explicitly
named repository files. They are not part of this hf_only_reference_validation schema-v2 entrypoint or
its authority; this README intentionally provides no historical commands.

## Unified server entrypoint

`hf_only_threshold_fit_server.py` is the complete Colab-neutral and
server-direct orchestration entrypoint for one frozen threshold-fit shard. It
requires a clean checkout at an explicit 40-hex revision plus disjoint absolute
scratch, cache, and output roots. It checks the registered GPU/VRAM floor and
available storage, builds the dedicated schema-v2 package from that exact Git
tree, delegates the frozen dependency installation and package trust boundary
to `experiment_execution_bootstrap.py`, downloads the runtime configuration's
exact `model_id` and `model_revision` into the supplied cache, and then invokes
the existing package entrypoint and formal runner. The runner remains the only
records writer.

Both `HF_TOKEN` and `CEG_WM_ROOT_KEY` are read from the process environment and
are never included in the machine-readable receipt. A server invocation is:

```bash
python scripts/experiment_execution/hf_only_threshold_fit_server.py \
  --repository-root /absolute/clean/CEG-WM \
  --expected-revision 0123456789abcdef0123456789abcdef01234567 \
  --scratch-root /absolute/scratch \
  --cache-root /absolute/cache \
  --output-root /absolute/output \
  --run-id hf-only-content-threshold-fit \
  --shard-index 0
```

The stdout JSON receipt identifies the result or diagnostic ZIP, its SHA-256,
the revision/run/shard, the package trust digests, and the frozen model
identity. This entrypoint does not fit or approve tau, unlock confirmation,
access held-out evaluation, or support scientific claims.
