---
title: "Threadblock API and Inner Loops"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#threadblock-api-and-inner-loops"
---

## [Threadblock API and Inner Loops](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#threadblock-api-and-inner-loops)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#threadblock-api-and-inner-loops "Permalink to this headline")

Much of the CUTLASS 3 GEMM hierarchy for mainloops and inner loops diverges
from that of CUTLASS 2.x.  With that also comes the introduction of the
`cutlass::gemm::collective` layer as a direct replacement and a superset
of the 2.x `cutlass::gemm::threadblock` layer. Going forward,
CUTLASS 3.x will discontinue new developments in the following namespaces.

- `cutlass::*::threadblock::*`
- `cutlass::*::warp::*`
- `cutlass::gemm::thread::*`
- `cutlass::arch::*` (except `barrier.h`)

`cutlass::gemm::collective`s are a superset of the threadblock layer where
all new mainloops will be developed. Users should look to the `CollectiveMma` type
if they wish to author custom mainloop code in the 3.x API.

Similarly, for the GEMM inner loops, `cute::MMA_Atom`s replace the
`gemm::warp` and `gemm::thread` layer code. Going forward, all new PTX instructions
and associated metadata development will occur directly inside [`cute/arch/*.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/) and [`cute/atom/*.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/).

The desired inner loop MMA iteration order and tiling can be achieved through careful
selection of the atom layout, value layout, and permutations of the `cute::TiledMma`.

For epilogues, the `cutlass::epilogue::collective` layer replaces `cutlass::threadblock::collective`.  However, the thread-level epilogue elementwise operations
in `cutlass::epilogue::thread` will continue to be used in 3.x kernels as well, albeit, with
a more idiomatic epilogue vectorization strategy.
[Example 50](https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu)
shows how to use 2.x epilogue thread operators with 3.0 API kernels.
