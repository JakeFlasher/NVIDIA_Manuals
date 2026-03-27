---
title: "API documentation"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_runtime.html#module-cutlass.cute.runtime"
---

## [API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute.runtime)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute.runtime "Permalink to this headline")

```
_`class`_`cutlass.cute.runtime.``_Pointer`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Pointer "Link to this definition")
```

Bases: `Pointer`

Runtime representation of a pointer that can inter-operate with various data structures,
including numpy arrays and device memory.

**Parameters:**
: - **pointer** (_int__or__pointer-like object_) – The pointer to the data
- **dtype** (_Type_) – Data type of the elements pointed to
- **mem_space** (__cute_ir.AddressSpace__,__optional_) – Memory space where the pointer resides, defaults to generic
- **assumed_align** (_int__,__optional_) – Assumed alignment of input pointer in bytes, defaults to None

**Variables:**
: - **_pointer** – The underlying pointer
- **_dtype** – Data type of the elements
- **_addr_space** – Memory space of the pointer
- **_assumed_align** – Alignment of the pointer in bytes
- **_desc** – C-type descriptor for the pointer
- **_c_pointer** – C-compatible pointer representation

```
`__init__`(
```

```
`size_in_bytes`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Pointer.size_in_bytes "Link to this definition")
```

```
_`property`_`mlir_type`_`:` `cutlass._mlir.ir.Type`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Pointer.mlir_type "Link to this definition")
```

```
_`property`_`dtype`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Pointer.dtype "Link to this definition")
```

```
_`property`_`memspace`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Pointer.memspace "Link to this definition")
```

```
`align`(
```

```
_`class`_`cutlass.cute.runtime.``_Tensor`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor "Link to this definition")
```

Bases: `Tensor`

```
`__init__`(
```

```
`load_dltensor`()[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.load_dltensor "Link to this definition")
```

Lazily load the DLTensorWrapper.

This function loads the DLTensorWrapper when needed,
avoiding overhead in the critical path of calling JIT functions.

```
`mark_layout_dynamic`(_`leading_dim``:` `int` `|` `None` `=` `None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.mark_layout_dynamic "Link to this definition")
```

Marks the tensor layout as dynamic based on the leading dimension.

**Parameters:**
: **leading_dim** (_int__,__optional_) – The leading dimension of the layout, defaults to None

When `leading_dim` is None, the leading dimension is deduced as follows.

1. If exactly one dimension has stride 1, that dimension is used.
2. If multiple dimensions have stride 1 but exactly one of them has size > 1,
that dimension is used.
3. If multiple dimensions have stride 1 but none or more than one has size > 1,
an error is raised.
4. If no dimension has stride 1, all strides remain dynamic.

When `leading_dim` is explicitly specified, marks the layout as dynamic while setting the
stride at `leading_dim` to 1. Also validates that the specified `leading_dim` is consistent
with the existing layout by checking that the corresponding stride of that dimension is 1.

Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

**Returns:**
: The tensor with dynamic layout

**Return type:**
: [_Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor "cutlass.cute.runtime._Tensor")

```
`mark_compact_shape_dynamic`(
```

Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

**Parameters:**
: - **mode** (_int_) – The mode of the compact shape, defaults to 0
- **stride_order** – Consistent with *torch.Tensor.dim_order*. Defaults to None.

Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
It starts from the outermost to the innermost dimension.
:type stride_order: tuple[int, …], optional
:param divisibility: The divisibility constraint for the compact shape, defaults to 1
:type divisibility: int, optional
:return: The tensor with dynamic compact shape
:rtype: _Tensor

If `stride_order` is not provided, the stride ordering will be automatically deduced from the layout.
Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
An error is raised if automatic deduction fails.

If `stride_order` is explicitly specified, it does the consistency check with the layout.

For example:
- Layout: (4,2):(1,4) has stride_order: (1,0) indicates the innermost dimension is 0(*4:1*), the outermost dimension is 1(*2:4*)
- Layout: (5,3,2,4):(3,1,15,30) has stride_order: (3,2,0,1) indicates the innermost dimension is 1(*3:1*), the outermost dimension is 3(*4:30*).

Using *torch.Tensor.dim_order()* to get the stride order of the torch tensor.
.. code-block:: python
a = torch.empty(3, 4)
t = cute.runtime.from_dlpack(a)
t = t.mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order())

```
_`property`_`element_type`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.element_type "Link to this definition")
```

```
_`property`_`memspace`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.memspace "Link to this definition")
```

```
_`property`_`size_in_bytes`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.size_in_bytes "Link to this definition")
```

```
_`property`_`mlir_type`_`:` `cutlass._mlir.ir.Type`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.mlir_type "Link to this definition")
```

```
_`property`_`iterator`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.iterator "Link to this definition")
```

```
_`property`_`layout`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.layout "Link to this definition")
```

```
_`property`_`shape`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.shape "Link to this definition")
```

```
_`property`_`stride`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.stride "Link to this definition")
```

```
_`property`_`leading_dim`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.leading_dim "Link to this definition")
```

Get the leading dimension of this Tensor.

**Returns:**
: The leading dimension index or indices

**Return type:**
: int or tuple or None

The return value depends on the tensor’s stride pattern:

- If a single leading dimension is found, returns an integer index
- If nested leading dimensions are found, returns a tuple of indices
- If no leading dimension is found, returns None

```
`fill`(_`value``:` `cutlass.cute.typing.Numeric`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.fill "Link to this definition")
```

```
_`property`_`data_ptr`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.data_ptr "Link to this definition")
```

```
_`property`_`dynamic_shapes_mask`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.dynamic_shapes_mask "Link to this definition")
```

Get the mask of dynamic shapes in the tensor.

```
_`property`_`dynamic_strides_mask`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._Tensor.dynamic_strides_mask "Link to this definition")
```

Get the mask of dynamic strides in the tensor.

```
`cutlass.cute.runtime.``_get_cute_type_str`(_`inp`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._get_cute_type_str "Link to this definition")
```

```
_`class`_`cutlass.cute.runtime.``_FakeTensor`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor "Link to this definition")
```

Bases: `Tensor`

Fake Tensor implementation as a placeholder.
It mimics the interface of Tensor, but does not hold real data or allow indexing.
Used for compilation or testing situations where only shape/type/layout information is needed.
All attempts to access or mutate data will raise errors.

```
`__init__`(
```

```
_`property`_`mlir_type`_`:` `cutlass._mlir.ir.Type`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.mlir_type "Link to this definition")
```

```
_`property`_`element_type`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.element_type "Link to this definition")
```

```
_`property`_`memspace`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.memspace "Link to this definition")
```

```
_`property`_`iterator`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.iterator "Link to this definition")
```

```
_`property`_`shape`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.shape "Link to this definition")
```

```
_`property`_`stride`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.stride "Link to this definition")
```

```
_`property`_`leading_dim`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.leading_dim "Link to this definition")
```

```
_`property`_`dynamic_shapes_mask`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.dynamic_shapes_mask "Link to this definition")
```

```
_`property`_`dynamic_strides_mask`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.dynamic_strides_mask "Link to this definition")
```

```
`fill`(_`value``:` `cutlass.cute.typing.Numeric`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor.fill "Link to this definition")
```

```
`cutlass.cute.runtime.``make_fake_compact_tensor`(
```

Create a fake tensor with the specified shape, element type, and a compact memory layout.

**Parameters:**
: - **dtype** (_Type__[__Numeric__]_) – Data type of the tensor elements.
- **shape** (_tuple__[__Union__[__int__,__SymInt__]__,__...__]_) – Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
- **stride_order** (_tuple__[__int__,__...__]__,__optional_) – Order in which strides (memory layout) are assigned to the tensor dimensions.
If None, the default layout is left-to-right order (known as column-major order for flatten layout).
Otherwise, it should be a permutation order of the dimension indices.
The mode with stride_order 0 is the fastest changing (leading) dimension, and N-1 is the slowest changing.
- **memspace** (_AddressSpace__,__optional_) – Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
- **assumed_align** (_int__,__optional_) – Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.
- **use_32bit_stride** (_bool__,__optional_) – Whether to use 32-bit stride for dynamic dimensions. If True and the total size of the
layout (cosize(layout)) fits within int32, then dynamic strides will use 32-bit integers for improved performance.
Only applies when dimensions are dynamic. Defaults to False.

**Returns:**
: An instance of a fake tensor with the given properties and compact layout.

**Return type:**
: [_FakeTensor](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

**Examples:**

```python
@cute.jit
def foo(x: cute.Tensor):
    ...

x = make_fake_compact_tensor(
    cutlass.Float32, (100, cute.sym_int32(divisibility=8)), stride_order=(1, 0)
)

# Compiled function will take a tensor with the type:
#   tensor<ptr<f32, generic> o (100,?{div=8}):(?{i32 div=8},1)>
compiled_foo = cute.compile(foo, x)

# Default stride order is left-to-right order (0, 1, ..., n-1)
y = make_fake_compact_tensor(cutlass.Float32, (8, 3, 2)) # y.stride == (1, 8, 24)
```

```
`cutlass.cute.runtime.``make_fake_tensor`(
```

Create a fake tensor with the specified element type, shape, and stride.

**Parameters:**
: - **dtype** (_Type__[__Numeric__]_) – Data type of the tensor elements.
- **shape** (_tuple__[__Union__[__int__,__SymInt__]__,__...__]_) – Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
- **stride** (_tuple__[__Union__[__int__,__SymInt__]__,__...__]_) – Stride of the tensor, consisting of static (int) or dynamic (SymInt) values.
- **memspace** (_AddressSpace__,__optional_) – Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
- **assumed_align** (_int__,__optional_) – Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.

**Returns:**
: An instance of a fake tensor with the given properties.

**Return type:**
: [_FakeTensor](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

```
_`class`_`cutlass.cute.runtime.``_FakeStream`(_`*`_, _`use_tvm_ffi_env_stream``:` `bool` `=` `False`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeStream "Link to this definition")
```

Bases: `object`

A fake stream that can be used as a placeholder for a stream in compilation.

When use_tvm_ffi_env_stream is True and the function is compiled with TVM-FFI,
the argument will be skipped from the function signature and we pass in
this value through the environment stream obtained from caller context
(e.g. torch.cuda.current_stream()).

```
`__init__`(_`*`_, _`use_tvm_ffi_env_stream``:` `bool` `=` `False`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeStream.__init__ "Link to this definition")
```

```
`use_tvm_ffi_env_stream`_`:` `bool`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime._FakeStream.use_tvm_ffi_env_stream "Link to this definition")
```

```
`cutlass.cute.runtime.``make_fake_stream`(_`*`_, _`use_tvm_ffi_env_stream``:` `bool` `=` `False`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime.make_fake_stream "Link to this definition")
```

Create a fake stream that can be used as a placeholder for a stream in compilation.

When use_tvm_ffi_env_stream is True and the function is compiled with TVM-FFI,
the argument will be skipped from the function signature and we pass in
this value through the environment stream obtained from caller context
(e.g. torch.cuda.current_stream()). This can speedup the calling process
since we no longer need to do stream query in python.

**Parameters:**
: **use_tvm_ffi_env_stream** (_bool_) – Whether to skip this parameter use environment stream instead.

```
`cutlass.cute.runtime.``from_dlpack`(
```

Convert from tensor object supporting __dlpack__() to a CuTe Tensor.

**Parameters:**
: - **tensor_dlpack** (_object_) – Tensor object that supports the DLPack protocol
- **assumed_align** (_int__,__optional_) – Assumed alignment of the tensor (bytes), defaults to None,
if None, will use the element size bytes as the assumed alignment.
- **use_32bit_stride** (_bool__,__optional_) – Whether to use 32-bit stride, defaults to False. When True, the dynamic
stride bitwidth will be set to 32 for small problem size (cosize(layout) <= Int32_max) for better performance.
This is only applied when the dimension is dynamic.
- **enable_tvm_ffi** (_bool__,__optional_) – Whether to enable TVM-FFI, defaults to False. When True, the tensor will be converted to
a TVM-FFI function compatible tensor.
- **force_tf32** (_bool__,__optional_) – Whether to force the element type to TFloat32 if the element type is Float32.

**Returns:**
: A CuTe Tensor object

**Return type:**
: Tensor

**Examples:**

```python
import torch
from cutlass.cute.runtime import from_dlpack
x = torch.randn(100, 100)
y = from_dlpack(x)
y.shape
# (100, 100)
type(y)
# <class 'cutlass.cute.Tensor'>
```

```
`cutlass.cute.runtime.``make_ptr`(
```

Create a pointer from a memory address

**Parameters:**
: - **dtype** (_Type__[__Numeric__]_) – Data type of the pointer elements
- **value** (_Union__[__int__,__ctypes._Pointer__]_) – Memory address as integer or ctypes pointer
- **mem_space** (_AddressSpace__,__optional_) – Memory address space, defaults to AddressSpace.generic
- **align_bytes** (_int__,__optional_) – Alignment in bytes, defaults to None

**Returns:**
: A pointer object

**Return type:**
: Pointer

```python
import numpy as np
import ctypes

from cutlass import Float32
from cutlass.cute.runtime import make_ptr

# Create a numpy array
a = np.random.randn(16, 32).astype(np.float32)

# Get pointer address as integer
ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Create pointer from address
y = make_ptr(cutlass.Float32, ptr_address)

# Check properties
print(y.element_type)
print(type(y))  # <class 'cutlass.cute.Pointer'>
```

```
`cutlass.cute.runtime.``nullptr`(
```

Create a null pointer which is useful for compilation

**Parameters:**
: - **dtype** (_Type__[__Numeric__]_) – Data type of the pointer elements
- **mem_space** (_AddressSpace__,__optional_) – Memory address space, defaults to AddressSpace.generic

**Returns:**
: A null pointer object

**Return type:**
: Pointer

```
_`class`_`cutlass.cute.runtime.``TensorAdapter`(_`arg`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime.TensorAdapter "Link to this definition")
```

Bases: `object`

Convert a DLPack protocol supported tensor/array to a cute tensor.

```
`__init__`(_`arg`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime.TensorAdapter.__init__ "Link to this definition")
```

```
`cutlass.cute.runtime.``find_runtime_libraries`(
```

Find the runtime libraries that needs to be available for loading modules.

**Parameters:**
: **enable_tvm_ffi** (_bool__,__optional_) – Whether to enable TVM-FFI.

**Returns:**
: A list of runtime libraries that needs to be available for loading modules.

**Return type:**
: list

```
`cutlass.cute.runtime.``load_module`(_`file_path``:` `str`_, _`*`_, _`enable_tvm_ffi``:` `bool` `=` `False`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.runtime.load_module "Link to this definition")
```

Load a module from a file path.

**Parameters:**
: - **file_path** (_str_) – The path to the module file
- **enable_tvm_ffi** (_bool__,__optional_) – Whether to enable TVM-FFI, defaults to True. When True, the module will be loaded as a TVM-FFI module.

**Returns:**
: A module object

**Return type:**
: module
