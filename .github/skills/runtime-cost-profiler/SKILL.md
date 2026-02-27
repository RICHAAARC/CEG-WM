---
name: runtime-cost-profiler
description: Profile runtime cost including latency, throughput, and memory for method and ablation comparisons. Use when users ask for efficiency benchmarking, compute budget analysis, or runtime performance audit.
---
# runtime-cost-profiler

Measure and report compute cost with reproducible methodology.

## Execute

1. Define profiling dimensions: latency, throughput, peak memory, and batch context.
2. Verify profiling setup controls model, precision, device, and input shape.
3. Capture run metadata so cost numbers are comparable across experiments.
4. Report variance and confidence boundaries for repeated runs.
5. Flag unfair or non-comparable profiling settings.

## Output Contract

- `issue`
- `cost_measurement_risk`
- `evidence`
- `severity`
- `recommended_fix`
