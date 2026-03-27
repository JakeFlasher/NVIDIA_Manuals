---
title: "Prepare Input File"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html#prepare-input-file"
---

### [Prepare Input File](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#prepare-input-file)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#prepare-input-file "Permalink to this headline")

Prepare a list of gemm problem definitions, in the form of a json list, to be evaluated by the heuristic. Here is a sample file with two problems:

```console
[
{
     "m" : 4096,
     "n" : 4096,
     "k" : 4096,
     "batch_count" : 1,
     "layout" : "tnn",
     "dtype_a" : "f16",
     "dtype_b" : "f16",
     "dtype_c" : "f16",
     "dtype_acc" : "f32",
     "dtype_d" : "f16",
     "beta" : 0.0,
     "use_fast_acc": false
},
{
     "m" : 4096,
     "n" : 4096,
     "k" : 4096,
     "batch_count" : 1,
     "layout": "tnn",
     "dtype_a" : "e5m2",
     "dtype_b" : "e5m2",
     "dtype_c" : "f32",
     "dtype_acc" : "f32",
     "dtype_d" : "e5m2",
     "beta" : 0.0,
     "use_fast_acc": true
}
]
```

Note: `use_fast_acc` only needs to be specified for FP8 kernels on SM90. Otherwise, it is ignored.
