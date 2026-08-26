# Minimal Architecture

The architecture separates research responsibilities by ownership. Content execution uses a deliberately small set of cross-namespace imports; the enforced boundary is that research code never imports the detachable governance layer.

```text
cegwm.shared        immutable key and numerical primitives
cegwm.method        embedding and scoring semantics
cegwm.runtime       model-specific execution adapters
cegwm.protocol      fixed data and decision contracts
experiments         executable content-chain flows
```

- `cegwm.shared`: key and immutable numerical primitives shared by method components.
- `cegwm.method`: watermark carrier, embedding, and blind scoring semantics.
- `cegwm.runtime`: model-specific execution and observation adapters; it does not own detection decisions.
- `cegwm.protocol`: records and protocol types independent of method implementation.
- `experiments.run_content_chain`: the executable content-chain flow that combines method, runtime, and protocol and writes explicit fixed-denominator results.

`governance` is an outer validation layer and is forbidden as a research-code dependency.
