# Colab Entrypoints

## Current authorized entrypoint

`semantic_texture_operational_preflight.ipynb` 是当前唯一授权执行 **Run all** 的入口。唯一 Drive 输入是 regular non-symlink `MyDrive/CEG-WM/models/inspyrenet/ckpt_base.pth`；每次运行写入新的 `semantic_texture_operational_preflight/exports/<fresh-run-id>/` namespace。

The following retained diagnostics are paused / not authorized and preserve producer-bound history: they 不读取、不迁移、不改写或混合 old records.

- `hf_only_detector_directional_validation.ipynb`: `0d4253ab2614c642563c566e6268565c337b503f`; `ceg_wm_hf_only_detector_directional_validation_binary32_budget_authority`; superseded `ceg_wm_hf_only_detector_directional_validation_initial_gate` is immutable partial evidence.
- `hf_transmission_diagnostic.ipynb`: `af1eea8f55086b583e3e5e4a02586959983db70b`; `ceg_wm_hf_transmission_diagnostic_server_execution`.
- `lf_transmission_diagnostic.ipynb`: `2337f9d7c773a6054d558108e31d07d35fbee42f`; `ceg_wm_lf_carrier_to_detector_transmission_diagnostic`; it uses the producer-bound whitening asset `a78c47184cf83ad351bb4442ebd31c218726de25` / `ceg_wm_lf_whitening_asset_fit_and_score_screening`.
- `lf_whitened_directional_validation.ipynb`: `51adb765cdddafcb4c65c357e899c77b4c9f36d2`; `ceg_wm_lf_whitened_directional_validation_prepared_feature_execution`.
- `lf_whitened_score_screening.ipynb`: `a78c47184cf83ad351bb4442ebd31c218726de25`; 1 个 non-scientific operational, 32 个 clean null-fit, and 8 个 paired raw-vs-whitened screening units. This is development-only: it fits no threshold or FPR and provides no candidate promotion.
- `qk_synchronization_write_diagnostic.ipynb`: `1c1ff50d56a81bccb8b1f738d5b5f2792251246d`; `ceg_wm_qk_vae_decoder_internal_operation_localization`; historic `ceg_wm_qk_runtime_failure_localization` and `ceg_wm_qk_synchronization_write_public_rgb8_diagnosis` records、diagnostics 与 intents 保持不可变，不读取、迁移、覆盖或混入. Its only authority is `1 operational / 0 scientific / 1 total / 1 attempt`.

`hf_transmission_diagnostic.ipynb`, `lf_transmission_diagnostic.ipynb`, and all retained notebooks other than the current entrypoint remain thin, output-free Notebook surfaces.
