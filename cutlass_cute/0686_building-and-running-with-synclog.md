---
title: "Building and Running with synclog"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#building-and-running-with-synclog"
---

### [Building and Running with synclog](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#building-and-running-with-synclog)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#building-and-running-with-synclog "Permalink to this headline")

After enabling `synclog`, build your CUTLASS example. For instance, to build example 54:

```console
$ cd examples/54_hopper_fp8_warp_specialized_gemm
$ make
```

Run the example, setting the profiling iteration count to 0 to ensure `synclog` information is printed only for the reference run:

```console
$ ./54_hopper_fp8_warp_specialized_gemm --iterations=0 &> synclog.txt
```
