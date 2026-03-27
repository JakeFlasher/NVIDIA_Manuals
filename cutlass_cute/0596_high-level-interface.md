---
title: "High-level interface"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#high-level-interface"
---

### [High-level interface](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute#high-level-interface)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/#high-level-interface "Permalink to this headline")

We’ll start with the kernel entry point `gemm_device` at the top of the file.

```c++
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
```

There are many template parameters, let’s quickly review them and then go into more depth on their uses.

- `ProblemShape`. The MxNxK problem shape of this matrix multiply.
- `CtaTiler`. A CuTe [tiler concept](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition-tilers) that determines how to extract a tile of data from the problem shape.
- `TA const* A`, `TB const* B`, `TC* C`. The types and pointers to the A, B, and C data, respectively.
- `AStride`, `BStride`, `CStride`. The layout strides corresponding to the `ProblemShape` for each A, B, and C.
- `ASmemLayout`, `BSmemLayout`, `CSmemLayout`. The layouts, if needed, of shared memory to use for staging A-data, B-data, and C-data within each CTA.
- `AThreadLayout`, `BThreadLayout`, `CThreadLayout`. The layouts of threads to be used in partitioning each stage.
- `Alpha alpha`, `Beta beta`. The types and values of the scalar constants to compute GEMM: `C = alpha * A * B + beta * C`.
