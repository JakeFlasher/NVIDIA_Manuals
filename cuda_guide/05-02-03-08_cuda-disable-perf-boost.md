---
title: "5.2.3.8. CUDA_DISABLE_PERF_BOOST"
section: "5.2.3.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-disable-perf-boost"
---

### [5.2.3.8. CUDA_DISABLE_PERF_BOOST](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-disable-perf-boost)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-disable-perf-boost "Permalink to this headline")

On Linux hosts, setting this environment variable to 1 prevents boosting the device performance state, instead pstate can be selected implicitly based on various heuristics.  This option can potentially be used to reduce power consumption, but may result in higher latency in certain scenarios due to dynamic performance state selection.

**Example**:

```bash
CUDA_DISABLE_PERF_BOOST=1 # perf boost disabled, Linux only.
CUDA_DISABLE_PERF_BOOST=0 # default behavior
```
