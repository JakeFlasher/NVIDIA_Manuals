---
title: "Bypass the DLPack Protocol"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#bypass-the-dlpack-protocol"
---

## [Bypass the DLPack Protocol](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#bypass-the-dlpack-protocol)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#bypass-the-dlpack-protocol "Permalink to this headline")

In certain scenarios, users may wish to bypass the DLPack protocol and invoke the JIT function directly.
This can be accomplished by creating a lightweight JIT wrapper around the existing JIT function,
utilizing `cute.ptr` and `cute.make_tensor` to pass pointers and construct tensors directly.

Typical use cases for bypassing DLPack include:
1. Users want to call the JIT function directly to avoid the overhead introduced by the DLPack protocol.
2. DLPack canonicalizes the stride of shape-1 dimensions to 1, which may result in incorrect alignment
propagation and affect memory access or performance.
3. DLPack may lack support for some narrow data types.

The following example illustrates how to bypass the DLPack protocol when invoking a JIT function.
Assume we have a pre-defined `TensorOpGemm` kernel whose JIT interface expects three
arguments of type `cute.Tensor`. To enable direct invocation without DLPack, we first define a JIT wrapper
function that accepts `cute.Pointer` types as parameters. Within this wrapper, we use `cute.make_tensor`
to construct tensors from the provided pointers, and then call the `TensorOpGemm` kernel as usual.

```python
@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):

    # Assume alignment of shape to call tensorop_gemm example
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch is row major
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    # TensorOpGemm is a pre-defined kernel from our example
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )

    tensor_op_gemm(mA, mB, mC)
```

To pass a PyTorch tensor to this new JIT wrapper, we retrieve the raw pointer from the PyTorch tensor
and create a `cute.Pointer` instance using `cute.make_ptr`.
This approach allows us to bypass the DLPack protocol entirely, avoiding its overhead and potential
issues with shape-1 dimension handling.

```python
a = torch.randn(
    m, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
b = torch.randn(
    n, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
c = torch.randn(
    n, m, l, dtype=torch.float16, device="cuda"
).permute(1, 2, 0)

# from cutlass.cute.runtime import make_ptr
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, m, n, k, l)
```
