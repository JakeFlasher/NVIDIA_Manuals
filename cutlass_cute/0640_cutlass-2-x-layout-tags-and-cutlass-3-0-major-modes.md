---
title: "CUTLASS 2.x layout tags and CUTLASS 3.0 major modes"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html#cutlass-2-x-layout-tags-and-cutlass-3-0-major-modes"
---

### [CUTLASS 2.x layout tags and CUTLASS 3.0 major modes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-2-x-layout-tags-and-cutlass-3-0-major-modes)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-2-x-layout-tags-and-cutlass-3-0-major-modes "Permalink to this headline")

CUTLASS 2.x and CUTLASS 3.0 use both
different wording and different types
to describe the permitted layouts
of GEMM’s input matrices A and B.

CUTLASS 3.0 does not use the terms “column major”
or “row major” to describe matrix layouts.
Starting with CUTLASS 3.0, adoption of CuTe allows us to decouple

- the coordinate mode order (logical shape) of layouts from
- the index space stride order of the backing storage.

In line with our switch to a conceptual GEMM hierarchy, we view the major modes not from a BLAS-3 perspective.
Rather, we divide the modes into two categories.

- “Inner modes” or “K-modes” are contracted over during the GEMM.
Therefore, they are not present in the output tensor.
- “Outer modes” or “MN-modes” are preserved in the output.

Now, instead of `RowMajor` or `ColumnMajor`, whose major stride depends on whether we are referring to the
A or the B matrix, we uniformly employ the “K major” or “MN major” terminology and enforce the convention of all tensors having the shape `[M/N, K, L]` regardless of which mode is major.  That is,

- the input matrix A has shape M x K,
- the input matrix B has shape N x K, and
- the input/output matrices C/D have shape M x N.

Note that this convention for B
differs from the BLAS’s GEMM interface,
which specifies that B has shape K x N.

CUTLASS 3.0 uses these names of the modes
to specify which mode of a matrix has stride 1.
For the matrix A,

- “M major” means that the matrix is stride 1
in the M mode, and
- “K major” means that the matrix is stride 1
in the K mode.

For the matrix B,

- “N major” means that the matrix is stride 1
in the N mode (which for B is mode 0,
because the convention is that B is N x K); and
- “K major” means that the matrix is stride 1
in the K mode (which for B is mode 1).

CUTLASS 2.x defines “layout tag” classes
`cutlass::layout::ColumnMajor` and `cutlass::layout::RowMajor`,
that live in the header file
[`cutlass/layout/matrix.h`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/matrix.h).
The interpretation of these layouts in GEMM
depends on whether they are applied
to the input matrix A or B. For the matrix A, “column major” means
that mode corresponding to M extent has stride 1,
and “row major” means that mode corresponding to K extent has stride 1.
This is the usual computer science definition
of column major and row major for a rank-2 array.
For the matrix B, the opposite holds:
“column major” means that mode corresponding to N extent has stride 1,
and “row major” means that mode corresponding to K extent has stride 1.

Using the convention of `[outer, inner, batch]` mode order for tensor logical shapes
avoids potential confusion with the meaning of column major and row major
changing depending on whether they are applied to A or B.

The table below summarizes our mode order convention and
mapping of 2.x layout tags to corresponding M-major, N-major, or K-major strides.

| Matrix | CUTLASS 2.x layout | 2.x Shape | Logical major mode | 3.x Shape/Stride | Major ordinal |
| --- | --- | --- | --- | --- | --- |
| A | `ColumnMajor` | M x K | M major | M x K x L | 0 (outer) |
| A | `RowMajor` | M x K | K major | M x K x L | 1 (inner) |
| B | `RowMajor` | K x N | N major | N x K x L | 0 (outer) |
| B | `ColumnMajor` | K x N | K major | N x K x L | 1 (inner) |
| C | `ColumnMajor` | M x N | M major | M x N x L | 0 (outer) |
| C | `RowMajor` | M x N | N major | M x N x L | 1 (inner) |

Notice that in CUTLASS 3.0, interpretation of layouts no longer changes based on
whether we are talking about the A or B matrix. M and N major inputs always have a
static size-1 stride in their 0th (outer) mode. Similarly, K major inputs
always contain the static size-1 stride in their 1st mode. This uniformity in stride order
allows us to represent tensor layouts much more cleanly and treat both A and B equally in our interfaces.
See for example the following snippet from our [`kernel/sm70_gemm.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm70_gemm.hpp)
for Ampere kernel schedules.

```c++
// Represent the full tensors
Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA); // (m,k,l)
Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); // (n,k,l)

// Get batch slice
Tensor mA_mk = mA_mkl(_,_,get<3>(blk_coord_mnkl)); // (m,k)
Tensor mB_nk = mB_nkl(_,_,get<3>(blk_coord_mnkl)); // (n,k)

// Slice to get the tiles for which this thread block is responsible
Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
```

As seem in this snippet, all input tensors have the logical shape `[outer, inner, batch]`,
and the strides could represent either outer or inner
(or any other complex hierarchical stride) major storage.
CuTe layouts always maintain the logical consistency of the coordinate spaces regardless of the strides.

By convention, in CUTLASS 3.0, we treat the M and N mode as the 0th mode,
and K mode as the 1st mode of the stride.
