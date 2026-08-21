# Minimal Architecture

The architecture separates research responsibilities without requiring empty implementations.

```text
cegwm.shared
    -> cegwm.method
        -> cegwm.runtime

cegwm.protocol
    -> experiments.stage_a <- cegwm.method + cegwm.runtime
```

- `cegwm.shared`: key and immutable numerical primitives shared by method components.
- `cegwm.method`: watermark carrier, embedding, and blind scoring semantics.
- `cegwm.runtime`: model-specific execution and observation adapters; it does not own detection decisions.
- `cegwm.protocol`: records and protocol types independent of method implementation.
- `experiments.stage_a`: the only layer that combines method, runtime, and protocol and writes Stage-A records.

`governance` is intentionally outside this graph and is forbidden as a research-code dependency. Directories are created when real code needs them; directory presence is never treated as implementation readiness.
