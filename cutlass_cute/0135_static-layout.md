---
title: "Static Layout"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html#static-layout"
---

## [Static Layout](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#static-layout)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#static-layout "Permalink to this headline")

When integrating with popular deep learning frameworks, one question is how to deal with the layout of the converted `cute.Tensor`.
For example, when converting a `torch.Tensor` to a `cute.Tensor`, the shape of the `torch.Tensor` is honored for the layout of
`cute.Tensor`.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime
```

In this example, we define a JIT function `foo` that takes a `cute.Tensor` as input and prints its layout. Note
that Python print is used to print the layout at compile time. This works fine for static layout whose value is known at
compile time.

Now let’s try to run the JIT function `foo` with different shapes of the input `torch.Tensor`.

```python
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
```

Here we first convert a 1D `torch.Tensor` with 3 elements to a `cute.Tensor` using `from_dlpack`. Then we compile
the JIT function `foo` with the converted `cute.Tensor` and call the compiled function.

```console
  tensor.layout: (3):(1)
  tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) =
( 1, 2, 3 )
```

It prints `(3):(1)` for the layout because the converted `cute.Tensor` has a static layout with shape `(3)` which
is the shape of the `a`.

Now if we call the compiled function with a different shape of the input `torch.Tensor`, it would result in an unexpected
result at runtime due to the mismatch of the type since `compiled_func` expects a `cute.Tensor` with layout `(3):(1)`
while `b` has shape `(5)`.

```python
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch
```

Following is the output which is unexpected due to the type mismatch.

```console
  tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) =
( 11, 12, 13 )
```

To fix that, we would have to trigger another code generation and compilation for the new shape for `b`.

```python
compiled_func_2 = cute.compile(foo, b_pack)  # This would trigger another compilation
compiled_func_2(b_pack)                      # ✅ Now this works fine
```

As shown in the example above, with the newly compiled `compiled_func_2`,  we can pass in `b_pack` to the compiled
JIT function `compiled_func_2`.

```console
  tensor.layout: (5):(1)
  tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) =
( 11, 12, 13, 14, 15 )
```

Now it recompiles and prints the values of `b` correctly.

It’s obvoius that we need distinct codes generated and compiled for different static layout. In this case, one for layout
`(3):(1)` and the other for layout `(5):(1)`.
