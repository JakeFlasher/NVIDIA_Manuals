---
title: "Thread-level GEMM API"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#thread-level-gemm-api"
---

### [Thread-level GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#thread-level-gemm-api)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#thread-level-gemm-api "Permalink to this headline")

Thread-level GEMM operations perform matrix multiply-accumulate on data held in registers. These target CUDA Cores exclusively.

_Concept._ Thread-level matrix multiply operations are function objects satisfying the following concept.

```c++
struct Mma {

  /// Shape of warp-level matrix operation (concept: GemmShape)
  struct Shape;

  /// Data type of multiplicand A (concept: numeric type)
  struct ElementA;

  /// Layout of multiplicand A (concept: Layout)
  struct LayoutA;

  /// Fragment object loaded from IteratorA (concept: Array<ElementA, ..>)
  struct FragmentA;

  /// Data type of multiplicand B (concept: numeric type)
  struct ElementB;

  /// Layout of multiplicand B (concept: Layout)
  struct LayoutB;

  /// Fragment object loaded from IteratorA (concept: Array<ElementB, ..>)
  struct FragmentB;

  /// Data type of accumulator matrix C (concept: numeric type)
  struct ElementC;

  /// Layout of accumulator matrix C (concept: Layout)
  struct LayoutC;

  /// Fragment object loaded from IteratorA (concept: Array<ElementC, ..>)
  struct FragmentC;

  //
  // Methods
  //

  /// Computes a matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    FragmentA const &A,
    FragmentB const &B,
    FragmentC const &C);
};
```

The CUTLASS thread-level GEMM template accepts the following template arguments.

```c++
namespace cutlass {
namespace gemm {
namespace thread {

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Concept: arch::OpMultiplyAdd or arch::Mma<>
  typename Operator = arch::OpMultiplyAdd,
  /// Used for partial specialization
  typename Enable = bool
>
struct Mma;

} // namespace thread
} // namespace gemm
} // namespace cutlass
```
