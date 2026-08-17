# Test Case Registry

This registry records the active lightweight project test surfaces. Formal profile selection remains governed by the project contract; retired delivery-family test files are preserved in Git ancestry rather than this active registry.

| family | path | kind | selected cases | scope |
| --- | --- | --- | ---: | --- |
| sd35_backend | `tests/unit/test_sd35_backend.py` | unit | 37 | CPU backend boundary and local-path prohibition |
| experiment_execution_support | `tests/unit/test_experiment_execution_support.py` | unit | 5 | neutral delivery/server helpers |
| notebook_delivery | `tests/unit/test_notebook_delivery.py` | unit | 1 | exact retained notebook inventory |
| development_worker_persistence | `tests/unit/test_development_worker_persistence.py` | quick | 39 | create-only persistence and replay |
| experiment_execution_delivery | `tests/unit/test_experiment_execution_delivery.py` | unit | 6 | HF-only threshold-fit delivery |
| hf_only_threshold_fit_delivery | `tests/unit/test_hf_only_threshold_fit_delivery.py::test_builder_uses_exact_threshold_fit_allowlist_and_is_deterministic` | unit | 1 | exact package allowlist |
| retained diagnostic delivery | `tests/unit/test_{hf_only_detector_directional_validation,hf_transmission_diagnostic,lf_transmission_diagnostic,lf_whitened_directional_validation,lf_whitened_score_screening,qk_synchronization_write_diagnostic}_delivery.py` | quick | 31 | active diagnostic entrypoint delivery |

The registered-addopts collection gate binds the exact active-suite count after the approved cleanup: 653 selected, 17 deselected, 670 total. The captured collection node log and SHA-256 are the authoritative node-level receipt for this revision.
