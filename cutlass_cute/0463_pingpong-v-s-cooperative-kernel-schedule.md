---
title: "Pingpong v.s. cooperative kernel schedule"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#pingpong-v-s-cooperative-kernel-schedule"
---

## [Pingpong v.s. cooperative kernel schedule](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#pingpong-v-s-cooperative-kernel-schedule)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#pingpong-v-s-cooperative-kernel-schedule "Permalink to this headline")

Similar to Hopper’s warp-group GEMM, SM120 GEMMs support both pingpong and cooperative kernel schedules. Pingpong kernel schedule has two groups of 4 MMA warps working on different output tiles, overlapping the mainloop and epilogue, while the cooperative kernel schedule has only one group of 8 MMA warps working on the same output tile. If `KernelScheduleAuto` is specified, `KernelTmaWarpSpecializedCooperative` will be selected by default.
