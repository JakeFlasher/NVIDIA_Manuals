---
title: "1.2 Tracing"
section: "1.2"
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html#tracing"
---

### [1.2 Tracing](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#tracing)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#tracing "Permalink to this headline")

The decorated function is executed once with _proxy_ arguments; overloaded
operators record every tensor operation that actually runs and produce a flat
trace that is lowered to intermediate representation (IR).

_Advantages_

- Near-zero compile latency, ideal for straight-line arithmetic.
- No need to parse Python source, so it supports many dynamic Python
features, and Python has many features.

_Disadvantages_

- Untaken branches vanish, so the generated kernel may be wrong for other
inputs.
- Loops are flattened to the iteration count observed during tracing.
- Data-dependent control-flow freezes to a single execution path.
