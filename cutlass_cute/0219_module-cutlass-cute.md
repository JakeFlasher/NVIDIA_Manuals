---
title: "cutlass.cute"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#module-cutlass.cute"
---

# [cutlass.cute](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api#module-cutlass.cute)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#module-cutlass.cute "Permalink to this headline")

```
_`class`_`cutlass.cute.``Swizzle`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Swizzle "Link to this definition")
```

Bases: `Value`

Swizzle is a transformation that permutes the elements of a layout.

Swizzles are used to rearrange data elements to improve memory access patterns
and computational efficiency.

Swizzle is defined by three parameters:
- MBase: The number of least-significant bits to keep constant
- BBits: The number of bits in the mask
- SShift: The distance to shift the mask

The mask is applied to the least-significant bits of the layout.

```console
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase is the number of least-sig bits to keep constant
                 ^-^       ^-^     BBits is the number of bits in the mask
                   ^---------^     SShift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
```

```
_`class`_`cutlass.cute.``struct`(_`cls`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct "Link to this definition")
```

Bases: `object`

Decorator to abstract C structure in Python DSL.

**Usage:**

```python
# Supports base_dsl scalar int/float elements, array and nested struct:
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32

@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16

# Supports alignment for its elements:
@cute.struct
class StorageB:
    a: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_a], 1024
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_b], 1024
    ]
    x: cute.struct.Align[cutlass.Int32, 16]
    compA: cute.struct.Align[complex, 16]

# Statically get size and alignment:
size = StorageB.__sizeof__()
align = StorageB.__alignof__()

# Allocate and referencing elements:
storage = allocator.allocate(StorageB)

storage.a[0] ...
storage.x ...
storage.compA.real ...
```

**Parameters:**
: **cls** – The struct class with annotations.

**Returns:**
: The decorated struct class.

```
_`class`_`_MemRangeMeta`(_`name`_, _`bases`_, _`dct`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta "Link to this definition")
```

Bases: `type`

A metaclass for creating MemRange classes.

This metaclass is used to dynamically create MemRange classes with specific
data types and sizes.

**Variables:**
: - **_dtype** – The data type of the MemRange.
- **_size** – The size of the MemRange.

```
`_dtype`_`=` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta._dtype "Link to this definition")
```

```
`_size`_`=` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta._size "Link to this definition")
```

```
_`property`_`size`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta.size "Link to this definition")
```

```
_`property`_`elem_width`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta.elem_width "Link to this definition")
```

```
_`property`_`size_in_bytes`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeMeta.size_in_bytes "Link to this definition")
```

```
_`class`_`MemRange`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct.MemRange "Link to this definition")
```

Bases: `object`

Defines a range of memory by *MemRange[T, size]*.

```
_`class`_`_MemRangeData`(_`dtype`_, _`size`_, _`base`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeData "Link to this definition")
```

Bases: `object`

Represents a range of memory.

**Parameters:**
: - **dtype** – The data type.
- **size** – The size of the memory range in bytes.
- **base** – The base address of the memory range.

```
`__init__`(_`dtype`_, _`size`_, _`base`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeData.__init__ "Link to this definition")
```

Initializes a new memory range.

**Parameters:**
: - **dtype** – The data type.
- **size** – Size of the memory range in bytes. A size of **0** is accepted, but in that
case the range can only be used for its address (e.g. as a partition marker).
- **base** – The base address of the memory range.

```
`data_ptr`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._MemRangeData.data_ptr "Link to this definition")
```

Returns start pointer to the data in this memory range.

**Returns:**
: A pointer to the start of the memory range.

**Raises:**
: **AssertionError** – If the size of the memory range is negative.

```
`get_tensor`(
```

Creates a tensor from the memory range.

**Parameters:**
: - **layout** – The layout of the tensor.
- **swizzle** – Optional swizzle pattern.
- **dtype** – Optional data type; defaults to the memory range’s data type if not specified.

**Returns:**
: A tensor representing the memory range.

**Raises:**
: - **TypeError** – If the layout is incompatible with the swizzle.
- **AssertionError** – If the size of the memory range is not greater than zero.

```
_`class`_`_AlignMeta`(_`name`_, _`bases`_, _`dct`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._AlignMeta "Link to this definition")
```

Bases: `type`

Aligns the given object by setting its alignment attribute.

**Parameters:**
: - **v** – The object to align. Must be a struct, MemRange, or a scalar type.
- **align** – The alignment value to set.

**Raises:**
: **TypeError** – If the object is not a struct, MemRange, or a scalar type.

**Variables:**
: - **_dtype** – The data type to be aligned.
- **_align** – The alignment of the data type.

```
`_dtype`_`=` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._AlignMeta._dtype "Link to this definition")
```

```
`_align`_`=` `None`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._AlignMeta._align "Link to this definition")
```

```
_`property`_`dtype`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._AlignMeta.dtype "Link to this definition")
```

```
_`property`_`align`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._AlignMeta.align "Link to this definition")
```

```
_`class`_`Align`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct.Align "Link to this definition")
```

Bases: `object`

Aligns the given type by *Align[T, alignment]*.

```
_`static`_`_is_scalar_type`(_`dtype`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct._is_scalar_type "Link to this definition")
```

Checks if the given type is a scalar numeric type.

**Parameters:**
: **dtype** – The type to check.

**Returns:**
: True if the type is a subclass of Numeric, False otherwise.

```
`__init__`(_`cls`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct.__init__ "Link to this definition")
```

Initializes a new struct decorator instance.

**Parameters:**
: **cls** – The class representing the structured data type.

**Raises:**
: **TypeError** – If the struct is empty.

```
`size_in_bytes`() → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct.size_in_bytes "Link to this definition")
```

Returns the size of the struct in bytes.

**Returns:**
: The size of the struct.

```
_`static`_`align_offset`(_`offset`_, _`align`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.struct.align_offset "Link to this definition")
```

Return the round-up offset up to the next multiple of align.

```
`cutlass.cute.``E`(_`mode``:` `int` `|` `List``[``int``]`_) → `ScaledBasis`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.E "Link to this definition")
```

Create a unit ScaledBasis element with the specified mode.

This function creates a ScaledBasis with value 1 and the given mode.
The mode represents the coordinate axis or dimension in the layout.

**Parameters:**
: **mode** (_Union__[__int__,__List__[__int__]__]_) – The mode (dimension) for the basis element, either a single integer or a list of integers

**Returns:**
: A ScaledBasis with value 1 and the specified mode

**Return type:**
: ScaledBasis

**Raises:**
: **TypeError** – If mode is not an integer or a list

**Examples:**

```python
# Create a basis element for the first dimension (mode 0)
e0 = E(0)

# Create a basis element for the second dimension (mode 1)
e1 = E(1)

# Create a basis element for a hierarchical dimension
e_hier = E([0, 1])
```

```
`cutlass.cute.``get_divisibility`(_`x``:` `int` `|` `cutlass.cute.typing.Integer`_) → `int`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.get_divisibility "Link to this definition")
```

```
`cutlass.cute.``is_static`(_`x``:` `Any`_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.is_static "Link to this definition")
```

Check if a value is statically known at compile time.

In CuTe, static values are those whose values are known at compile time,
as opposed to dynamic values which are only known at runtime.

This function checks if a value is static by recursively traversing its type hierarchy
and checking if all components are static.

Static values include:
- Python literals (bool, int, float, None)
- Static ScaledBasis objects
- Static ComposedLayout objects
- Static IR types
- Tuples containing only static values

Dynamic values include:
- Numeric objects (representing runtime values)
- Dynamic expressions
- Any tuple containing dynamic values

**Parameters:**
: **x** (_Any_) – The value to check

**Returns:**
: True if the value is static, False otherwise

**Return type:**
: bool

**Raises:**
: **TypeError** – If an unsupported type is provided

```
`cutlass.cute.``has_underscore`(_`a``:` `cutlass.cute.typing.XTuple`_) → `bool`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.has_underscore "Link to this definition")
```

```
`cutlass.cute.``pretty_str`(_`arg`_) → `str`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.pretty_str "Link to this definition")
```

Constructs a concise readable pretty string.

```
`cutlass.cute.``printf`(_`*``args`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.printf "Link to this definition")
```

Print one or more values with optional formatting.

This function provides printf-style formatted printing capabilities. It can print values directly
or format them using C-style format strings. The function supports printing various types including
layouts, numeric values, tensors, and other CuTe objects.

The function accepts either:
1. A list of values to print directly
2. A format string followed by values to format

**Parameters:**
: - **args** (_Any_) – Variable length argument list containing either:
- One or more values to print directly
- A format string followed by values to format
- **loc** (_Optional__[__Location__]_) – Source location information for debugging, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for code generation, defaults to None

**Raises:**
: - **ValueError** – If no arguments are provided
- **TypeError** – If an unsupported argument type is passed

**Examples:**

Direct printing of values:

```python
a = cute.make_layout(shape=(10, 10), stride=(10, 1))
b = cutlass.Float32(1.234)
cute.printf(a, b)  # Prints values directly
```

Formatted printing:

```python
# Using format string with generic format specifiers
cute.printf("a={}, b={}", a, b)

# Using format string with C-style format specifiers
cute.printf("a={}, b=%.2f", a, b)
```

```
`cutlass.cute.``front`(_`input`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.front "Link to this definition")
```

Recursively get the first element of input.

This function traverses a hierarchical structure (like a layout or tensor)
and returns the first element at the deepest level. It’s particularly useful
for accessing the first stride value in a layout to determine properties like
majorness.

**Parameters:**
: - **input** (_Union__[__Tensor__,__Layout__,__Stride__]_) – The hierarchical structure to traverse
- **loc** (_source location__,__optional_) – Source location where it’s called, defaults to None
- **ip** (_insertion pointer__,__optional_) – Insertion pointer for IR generation, defaults to None

**Returns:**
: The first element at the deepest level of the input structure

**Return type:**
: Union[int, float, bool, ir.Value]

```
`cutlass.cute.``is_major`(
```

Check whether a mode in stride is the major mode.

```
`cutlass.cute.``assume`(_`src`_, _`divby``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.assume "Link to this definition")
```

```
`cutlass.cute.``make_swizzle`(_`b`_, _`m`_, _`s`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_swizzle "Link to this definition")
```

```
`cutlass.cute.``static`(_`value`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.static "Link to this definition")
```

```
`cutlass.cute.``get_leaves`(_`value`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.get_leaves "Link to this definition")
```

```
`cutlass.cute.``depth`(
```

Returns the depth (nesting level) of a tuple, layout, or tensor.

The depth of a tuple is the maximum depth of its elements plus 1.
For an empty tuple, the depth is 1. For layouts and tensors, the depth
is determined by the depth of their shape. For non-tuple values (e.g., integers),
the depth is considered 0.

**Parameters:**
: **a** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__,__Any__]_) – The object whose depth is to be determined

**Returns:**
: The depth of the input object

**Return type:**
: int

**Example:**

```python
>>> depth(1)
0
>>> depth((1, 2))
1
>>> depth(((1, 2), (3, 4)))
2
```

```
`cutlass.cute.``rank`(
```

Returns the rank (dimensionality) of a tuple, layout, or tensor.

The rank of a tuple is its length. For layouts and tensors, the rank is
determined by the rank of their shape. For non-tuple values (e.g., integers),
the rank is considered 1 for convenience.

**Parameters:**
: **a** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__,__Any__]_) – The object whose rank is to be determined

**Returns:**
: The rank of the input object

**Return type:**
: int

This function is used in layout algebra to determine the dimensionality
of tensors and layouts for operations like slicing and evaluation.

```
`cutlass.cute.``is_congruent`(
```

Returns whether a is congruent to b.

Congruence is an equivalence relation between hierarchical structures.

Two objects are congruent if:
* They have the same rank, AND
* They are both non-tuple values, OR
* They are both tuples AND all corresponding elements are congruent.

Congruence requires type matching at each level – scalar values match with
scalar values, and tuples match with tuples of the same rank.

**Parameters:**
: - **a** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__]_) – First object to compare
- **b** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__]_) – Second object to compare

**Returns:**
: True if a and b are congruent, False otherwise

**Return type:**
: bool

```
`cutlass.cute.``is_weakly_congruent`(
```

Returns whether a is weakly congruent to b.

Weak congruence is a partial order on hierarchical structures.

Object X is weakly congruent to object Y if:
* X is a non-tuple value, OR
* X and Y are both tuples of the same rank AND all corresponding elements are weakly congruent.

Weak congruence allows scalar values to match with tuples, making it useful
for determining whether an object has a hierarchical structure “up to” another.

**Parameters:**
: - **a** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__]_) – First object to compare
- **b** (_Union__[__XTuple__,__Layout__,__ComposedLayout__,__Tensor__]_) – Second object to compare

**Returns:**
: True if a and b are weakly congruent, False otherwise

**Return type:**
: bool

```
`cutlass.cute.``get`(_`input`_, _`mode``:` `List``[``int``]`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.get "Link to this definition")
```

Extract a specific element or sub-layout from a layout or tuple.

This function recursively traverses the input according to the mode indices,
extracting the element at the specified path. For layouts, this operation
corresponds to extracting a specific sub-layout.

**Parameters:**
: - **input** (_Layout__,__ComposedLayout__,__tuple_) – The input layout or tuple to extract from
- **mode** (_List__[__int__]_) – Indices specifying the path to traverse for extraction
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The extracted element or sub-layout

**Return type:**
: Layout, ComposedLayout, or element type

**Raises:**
: - **ValueError** – If any index in mode is out of range
- **TypeError** – If mode contains non-integer elements or if input has unsupported type

**Postcondition:**
: `get(t, mode=find(x,t)) == x if find(x,t) != None else True`

**Examples:**

```python
layout = make_layout(((4, 8), (16, 1), 8), stride=((1, 4), (32, 0), 512))
sub_layout = get(layout, mode=[0, 1])   # 8:4
sub_layout = get(layout, mode=[1])      # (16, 1):(32, 0)
```

```
`cutlass.cute.``select`(_`input`_, _`mode``:` `List``[``int``]`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.select "Link to this definition")
```

Select modes from input.

**Parameters:**
: - **input** (_Layout__,__ComposedLayout__,__tuple_) – Input to select from
- **mode** (_List__[__int__]_) – Indices specifying which dimensions or elements to select
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: A new instance with selected dimensions/elements

**Return type:**
: Layout, ComposedLayout, tuple

**Raises:**
: - **ValueError** – If any index in mode is out of range
- **TypeError** – If the input type is invalid

**Examples:**

```python
# Select specific dimensions from a layout
layout = make_layout((4, 8, 16), stride=(32, 4, 1))
selected = select(layout, mode=[0, 2])  # Select mode 0 and mode 2
# Result: (4, 16):(32, 1)

# Select elements from a tuple
t = (1, 2, 3, 4, 5)
selected = select(t, mode=[0, 2, 4])  # Select mode 0, mode 2, and mode 4
# Result: (1, 3, 5)
```

```
`cutlass.cute.``group_modes`(
```

Group modes of a hierarchical tuple or layout into a single mode.

This function groups a range of modes from the input object into a single mode,
creating a hierarchical structure. For tuples, it creates a nested tuple containing
the specified range of elements. For layouts and other CuTe objects, it creates
a hierarchical representation where the specified modes are grouped together.

**Parameters:**
: - **input** (_Layout__,__ComposedLayout__,__tuple__,__Shape__,__Stride__,__etc._) – Input object to group modes from (layout, tuple, etc.)
- **beg** (_int_) – Beginning index of the range to group (inclusive)
- **end** (_int_) – Ending index of the range to group (exclusive)
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: A new object with the specified modes grouped

**Return type:**
: Same type as input with modified structure

**Examples:**

```python
# Group modes in a tuple
t = (2, 3, 4, 5)
grouped = group_modes(t, 1, 3)  # (2, (3, 4), 5)

# Group modes in a layout
layout = make_layout((2, 3, 4, 5))
grouped_layout = group_modes(layout, 1, 3)  # Layout with shape (2, (3, 4), 5)

# Group modes in a shape
shape = make_shape(2, 3, 4, 5)
grouped_shape = group_modes(shape, 0, 2)  # Shape ((2, 3), 4, 5)
```

```
`cutlass.cute.``slice_`(_`src`_, _`coord``:` `cutlass.cute.typing.Coord`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.slice_ "Link to this definition")
```

Perform a slice operation on a source object using the given coordinate.

This function implements CuTe’s slicing operation which extracts a subset of elements
from a source object (tensor, layout, etc.) based on a coordinate pattern. The slice
operation preserves the structure of the source while selecting specific elements.

**Parameters:**
: - **src** (_Union__[__Tensor__,__Layout__,__IntTuple__,__Value__]_) – Source object to be sliced (tensor, layout, tuple, etc.)
- **coord** (_Coord_) – Coordinate pattern specifying which elements to select
- **loc** (_Optional__[__Location__]_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for IR generation, defaults to None

**Returns:**
: A new object containing the sliced elements

**Return type:**
: Union[Tensor, Layout, IntTuple, tuple]

**Raises:**
: **ValueError** – If the coordinate pattern is incompatible with source

**Examples:**

```python
# Layout slicing
layout = make_layout((4,4))

# Select 1st index of first mode and keep all elements in second mode
sub_layout = slice_(layout, (1, None))
```

```python
# Basic tensor slicing
tensor = make_tensor(...)           # Create a 2D tensor

# Select 1st index of first mode and keep all elements in second mode
sliced = slice_(tensor, (1, None))
```

```python
# Select 2nd index of second mode and keep all elements in first mode
sliced = slice_(tensor, (None, 2))
```

> **Note**
>
> - *None* represents keeping all elements in that mode
> - Slicing preserves the layout/structure of the original object
> - Can be used for:
> * Extracting sub-tensors/sub-layouts
> * Creating views into data
> * Selecting specific patterns of elements

```
`cutlass.cute.``prepend`(
```

Extend input to rank up_to_rank by prepending elem in front of input.

This function extends the input object by prepending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

**Parameters:**
: - **input** (_Union__[__Shape__,__Stride__,__Coord__,__IntTuple__,__Tile__,__Layout__,__ComposedLayout__,__Tensor__]_) – Source to be prepended to
- **elem** (_Union__[__Shape__,__Stride__,__Coord__,__IntTuple__,__Tile__,__Layout__]_) – Element to prepend to input
- **up_to_rank** (_Union__[__None__,__int__]__,__optional_) – The target rank after extension, defaults to None
- **loc** (_Optional__[__Location__]_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point, defaults to None

**Returns:**
: The extended result with prepended elements

**Return type:**
: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

**Raises:**
: - **ValueError** – If up_to_rank is less than input’s current rank
- **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Prepend to a Shape
shape = (4,4)
prepend(shape, 2)                   # Returns (2,4,4)

# Prepend to a Layout
layout = make_layout((8,8))
prepend(layout, make_layout((2,)))  # Returns (2,8,8):(1,1,8)

# Prepend with target rank
coord = (1,1)
prepend(coord, 0, up_to_rank=4)     # Returns (0,0,1,1)
```

```
`cutlass.cute.``append`(
```

Extend input to rank up_to_rank by appending elem to the end of input.

This function extends the input object by appending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

**Parameters:**
: - **input** (_Union__[__Shape__,__Stride__,__Coord__,__IntTuple__,__Tile__,__Layout__,__ComposedLayout__,__Tensor__]_) – Source to be appended to
- **elem** (_Union__[__Shape__,__Stride__,__Coord__,__IntTuple__,__Tile__,__Layout__]_) – Element to append to input
- **up_to_rank** (_Union__[__None__,__int__]__,__optional_) – The target rank after extension, defaults to None
- **loc** (_Optional__[__Location__]_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point, defaults to None

**Returns:**
: The extended result with appended elements

**Return type:**
: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]

**Raises:**
: - **ValueError** – If up_to_rank is less than input’s current rank
- **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Append to a Shape
shape = (4,4)
append(shape, 2)                   # Returns (4,4,2)

# Append to a Layout
layout = make_layout((8,8))
append(layout, make_layout((2,)))  # Returns (8,8,2):(1,8,1)

# Append with target rank
coord = (1,1)
append(coord, 0, up_to_rank=4)     # Returns (1,1,0,0)
```

> **Note**
>
> - The function preserves the structure of the input while extending it
> - Can be used to extend tensors, layouts, shapes and other CuTe types
> - When up_to_rank is specified, fills remaining positions with elem
> - Useful for tensor reshaping and layout transformations

```
`cutlass.cute.``prepend_ones`(
```

```
`cutlass.cute.``append_ones`(_`t`_, _`up_to_rank``:` `int` `|` `None` `=` `None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.append_ones "Link to this definition")
```

```
`cutlass.cute.``repeat_as_tuple`(_`x`_, _`n`_) → `tuple`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.repeat_as_tuple "Link to this definition")
```

Creates a tuple with x repeated n times.

This function creates a tuple by repeating the input value x n times.

**Parameters:**
: - **x** (_Any_) – The value to repeat
- **n** (_int_) – Number of times to repeat x

**Returns:**
: A tuple containing x repeated n times

**Return type:**
: tuple

**Examples:**

```python
repeat_as_tuple(1, 1)     # Returns (1,)
repeat_as_tuple(1, 3)     # Returns (1, 1, 1)
repeat_as_tuple(None, 4)  # Returns (None, None, None, None)
```

```
`cutlass.cute.``repeat`(_`x`_, _`n`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.repeat "Link to this definition")
```

Creates an object by repeating x n times.

This function creates an object by repeating the input value x n times.
If n=1, returns x directly, otherwise returns a tuple of x repeated n times.

**Parameters:**
: - **x** (_Any_) – The value to repeat
- **n** (_int_) – Number of times to repeat x

**Returns:**
: x if n=1, otherwise a tuple containing x repeated n times

**Return type:**
: Union[Any, tuple]

**Raises:**
: **ValueError** – If n is less than 1

**Examples:**

```python
repeat(1, 1)     # Returns 1
repeat(1, 3)     # Returns (1, 1, 1)
repeat(None, 4)  # Returns (None, None, None, None)
```

```
`cutlass.cute.``repeat_like`(_`x`_, _`target`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.repeat_like "Link to this definition")
```

Creates an object congruent to target and filled with x.

This function recursively creates a nested tuple structure that matches the structure
of the target, with each leaf node filled with the value x.

**Parameters:**
: - **x** (_Any_) – The value to fill the resulting structure with
- **target** (_Union__[__tuple__,__Any__]_) – The structure to mimic

**Returns:**
: A structure matching target but filled with x

**Return type:**
: Union[tuple, Any]

**Examples:**

```python
repeat_like(0, (1, 2, 3))      # Returns (0, 0, 0)
repeat_like(1, ((1, 2), 3))    # Returns ((1, 1), 1)
repeat_like(2, 5)              # Returns 2
```

```
`cutlass.cute.``flatten`(_`a`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.flatten "Link to this definition")
```

Flattens a CuTe data structure into a simpler form.

For tuples, this function flattens the structure into a single-level tuple.
For layouts, it returns a new layout with flattened shape and stride.
For tensors, it returns a new tensor with flattened layout.
For other types, it returns the input unchanged.

**Parameters:**
: **a** (_Union__[__IntTuple__,__Coord__,__Shape__,__Stride__,__Layout__,__Tensor__]_) – The structure to flatten

**Returns:**
: The flattened structure

**Return type:**
: Union[tuple, Any]

**Examples:**

```python
flatten((1, 2, 3))                      # Returns (1, 2, 3)
flatten(((1, 2), (3, 4)))               # Returns (1, 2, 3, 4)
flatten(5)                              # Returns 5
flatten(Layout(shape, stride))          # Returns Layout(flatten(shape), flatten(stride))
flatten(Tensor(layout))                 # Returns Tensor(flatten(layout))
```

```
`cutlass.cute.``filter_zeros`(_`input`_, _`*`_, _`target_profile``=``None`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.filter_zeros "Link to this definition")
```

Filter out zeros from a layout or tensor.

This function removes zero-stride dimensions from a layout or tensor.
Refer to [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
for more layout algebra operations.

**Parameters:**
: - **input** (_Layout__or__Tensor_) – The input layout or tensor to filter
- **target_profile** (_Stride__,__optional_) – Target stride profile for the filtered result, defaults to None
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The filtered layout or tensor with zeros removed

**Return type:**
: Layout or Tensor

**Raises:**
: **TypeError** – If input is not a Layout or Tensor

```
`cutlass.cute.``filter`(_`input`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.filter "Link to this definition")
```

Filter a layout or tensor.

This function filters a layout or tensor according to CuTe’s filtering rules.

**Parameters:**
: - **input** (_Layout__or__Tensor_) – The input layout or tensor to filter
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The filtered layout or tensor

**Return type:**
: Layout or Tensor

**Raises:**
: **TypeError** – If input is not a Layout or Tensor

```
`cutlass.cute.``size`(
```

Return size of domain of layout or tensor.

Computes the size (number of elements) in the domain of a layout or tensor.
For layouts, this corresponds to the shape of the coordinate space.
See [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)
for more details on layout domains.

**Parameters:**
: - **a** (_IntTuple__,__Shape__,__Layout__,__ComposedLayout__or__Tensor_) – The input object whose size to compute
- **mode** (_list__of__int__,__optional_) – List of mode(s) for size calculation. If empty, computes total size, defaults to []
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: Static size of layout or tensor if static, otherwise a Value

**Return type:**
: int or Value

**Raises:**
: **ValueError** – If mode contains non-integer elements

```
`cutlass.cute.``shape_div`(
```

Perform element-wise division of shapes.

This function performs element-wise division between two shapes.

**Parameters:**
: - **lhs** (_Shape_) – Left-hand side shape
- **rhs** (_Shape_) – Right-hand side shape
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The result of element-wise division

**Return type:**
: Shape

```
`cutlass.cute.``ceil_div`(
```

Compute the ceiling division of a target shape by a tiling specification.

This function computes the number of tiles required to cover the target domain.
It is equivalent to the second mode of *zipped_divide(input, tiler)*.

**Parameters:**
: - **input** (_Shape_) – A tuple of integers representing the dimensions of the target domain.
- **tiler** (_Union__[__Layout__,__Shape__,__Tile__]_) – The tiling specification.
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions.

**Returns:**
: A tuple of integers representing the number of tiles required along each dimension,
i.e. the result of the ceiling division of the input dimensions by the tiler dimensions.

**Return type:**
: Shape

Example:

```python
import cutlass.cute as cute
@cute.jit
def foo():
    input = (10, 6)
    tiler = (3, 4)
    result = cute.ceil_div(input, tiler)
    print(result)  # Outputs: (4, 2)
```

```
`cutlass.cute.``round_up`(
```

Rounds up elements of a using elements of b.

```
`cutlass.cute.``make_layout`(
```

Create a CuTe Layout object from shape and optional stride information.

A Layout in CuTe represents the mapping between logical and physical coordinates of a tensor.
This function creates a Layout object that defines how tensor elements are arranged in memory.

**Parameters:**
: - **shape** (_Shape_) – Shape of the layout defining the size of each mode
- **stride** (_Union__[__Stride__,__None__]_) – Optional stride values for each mode, defaults to None
- **loc** (_Optional__[__Location__]_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for IR generation, defaults to None

**Returns:**
: A new Layout object with the specified shape and stride

**Return type:**
: Layout

**Examples:**

```python
# Create a 2D compact left-most layout with shape (4,4)
layout = make_layout((4,4))                     # compact left-most layout

# Create a left-most layout with custom strides
layout = make_layout((4,4), stride=(1,4))       # left-most layout with strides (1,4)

# Create a layout for a 3D tensor
layout = make_layout((32,16,8))                 # left-most layout

# Create a layout with custom strides
layout = make_layout((2,2,2), stride=(4,1,2))   # layout with strides (4,1,2)
```

> **Note**
>
> - If stride is not provided, a default compact left-most stride is computed based on the shape
> - The resulting layout maps logical coordinates to physical memory locations
> - The layout object can be used for tensor creation and memory access patterns
> - Strides can be used to implement:
> * Row-major vs column-major layouts
> * Padding and alignment
> * Blocked/tiled memory arrangements
> * Interleaved data formats
> - Stride is keyword only argument to improve readability, e.g.
> * make_layout((3,4), (1,4)) can be confusing with make_layout(((3,4), (1,4)))
> * make_layout((3,4), stride=(1,4)) is more readable

```
`cutlass.cute.``make_identity_layout`(
```

Create an identity layout with the given shape.

An identity layout maps logical coordinates directly to themselves without any transformation.
This is equivalent to a layout with stride (1@0,1@1,…,1@(N-1)).

**Parameters:**
: - **shape** (_Shape_) – The shape of the layout
- **loc** (_Optional__[__Location__]_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for IR generation, defaults to None

**Returns:**
: A new identity Layout object with the specified shape

**Return type:**
: Layout

**Examples:**

```python
# Create a 2D identity layout with shape (4,4)
layout = make_identity_layout((4,4))     # stride=(1@0,1@1)

# Create a 3D identity layout
layout = make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)
```

> **Note**
>
> - An identity layout is a special case where each coordinate maps to itself
> - Useful for direct coordinate mapping without any transformation

```
`cutlass.cute.``make_ordered_layout`(
```

Create a layout with a specific ordering of dimensions.

This function creates a layout where the dimensions are ordered according to the
specified order parameter, allowing for custom dimension ordering in the layout.

**Parameters:**
: - **shape** (_Shape_) – The shape of the layout
- **order** (_Shape_) – The ordering of dimensions
- **loc** (_Optional__[__Location__]_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for IR generation, defaults to None

**Returns:**
: A new Layout object with the specified shape and dimension ordering

**Return type:**
: Layout

**Examples:**

```python
# Create a row-major layout
layout = make_ordered_layout((4,4), order=(1,0))

# Create a column-major layout
layout = make_ordered_layout((4,4), order=(0,1))         # stride=(1,4)

# Create a layout with custom dimension ordering for a 3D tensor
layout = make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)
```

> **Note**
>
> - The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
> - For a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
> - The length of order must match the rank of the shape

```
`cutlass.cute.``make_layout_like`(
```

```
`cutlass.cute.``make_composed_layout`(
```

Create a composed layout by composing an inner transformation with an outer layout.

A composed layout applies a sequence of transformations
to coordinates. The composition is defined as (inner ∘ offset ∘ outer), where the operations
are applied from right to left.

**Parameters:**
: - **inner** (_Union__[__Layout__,_[_Swizzle_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Swizzle "cutlass.cute.Swizzle")_]_) – The inner transformation (can be a Layout or Swizzle)
- **offset** (_IntTuple_) – An integral offset applied between transformations
- **outer** (_Layout_) – The outer (right-most) layout that is applied first
- **loc** (_Optional__[__Location__]_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for IR generation, defaults to None

**Returns:**
: A new ComposedLayout representing the composition

**Return type:**
: ComposedLayout

**Examples:**

```python
# Create a basic layout
inner = make_layout(...)
outer = make_layout((4,4), stride=(E(0), E(1)))

# Create a composed layout with an offset
composed = make_composed_layout(inner, (2,0), outer)
```

> **Note**
>
> - The composition applies transformations in the order: outer → offset → inner
> - The stride divisibility condition must be satisfied for valid composition
> - Certain compositions (like Swizzle with scaled basis) are invalid and will raise errors
> - Composed layouts inherit many properties from the outer layout

```
`cutlass.cute.``cosize`(
```

Return size of codomain of layout or tensor. Return static value if type is static.

For a layout `L = S:D` where `S` is the shape and `D` is the stride, the codomain size is the
minimum size needed to store all possible offsets generated by the layout. This is calculated
by taking the maximum offset plus 1.

**For example, given a layout `L = (4,(3,2)):(2,(8,1))`:**

  - Shape `S = (4,(3,2))`
  - Stride `D = (2,(8,1))`
  - Maximum offset = `2*(4-1) + 8*(3-1) + 1*(2-1) = 6 + 16 + 1 = 23`
  - Therefore `cosize(L) = 24`

**Examples:**

```python
L = cute.make_layout((4,(3,2)), stride=(2,(8,1))) # L = (4,(3,2)):(2,(8,1))
print(cute.cosize(L))  # => 24
```

**Parameters:**
: - **a** (_Union__[__Layout__,__ComposedLayout__,__Tensor__]_) – Layout, ComposedLayout, or Tensor object
- **mode** (_List__[__int__]__,__optional_) – List of mode(s) for cosize calculation. If empty, calculates over all modes.
If specified, calculates cosize only for the given modes.
- **loc** (_optional_) – Location information for diagnostics, defaults to None
- **ip** (_optional_) – Instruction pointer for diagnostics, defaults to None

**Returns:**
: Static size of layout or tensor (fast fold) if static, or a dynamic Value

**Return type:**
: Union[int, Value]

```
`cutlass.cute.``size_in_bytes`(
```

Calculate the size in bytes based on its data type and layout. The result is rounded up to the nearest byte.

**Parameters:**
: - **dtype** (_Type__[__Numeric__]_) – The DSL numeric data type
- **layout** (_Layout__,__optional_) – The layout of the elements. If None, the function returns 0
- **loc** (_optional_) – Location information for diagnostics, defaults to None
- **ip** (_optional_) – Instruction pointer for diagnostics, defaults to None

**Returns:**
: The total size in bytes. Returns 0 if the layout is None

**Return type:**
: int

```
`cutlass.cute.``coalesce`(
```

```
`cutlass.cute.``crd2idx`(
```

Convert a multi-dimensional coordinate into a value using the specified layout.

This function computes the inner product of the flattened coordinate and stride:

> index = sum(flatten(coord)[i] * flatten(stride)[i] for i in range(len(coord)))

**Parameters:**
: - **coord** (_Coord_) – A tuple or list representing the multi-dimensional coordinate
(e.g., (i, j) for a 2D layout).
- **layout** (_Layout__or__ComposedLayout_) – A layout object that defines the memory storage layout, including shape and stride,
used to compute the inner product.
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions.

**Returns:**
: The result of applying the layout transformation to the provided coordinate.

**Return type:**
: Any type that the layout maps to

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    L = cute.make_layout((5, 4), stride=(4, 1))
    idx = cute.crd2idx((2, 3), L)
    # Computed as: 2 * 4 + 3 = 11
    print(idx)
foo()  # Expected output: 11
```

```
`cutlass.cute.``idx2crd`(_`idx`_, _`shape`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.idx2crd "Link to this definition")
```

Convert a linear index back into a multi-dimensional coordinate using the specified layout.

Mapping from a linear index to the corresponding multi-dimensional coordinate in the layout’s coordinate space.
It essentially “unfolds” a linear index into its constituent coordinate components.

**Parameters:**
: - **idx** (_: int/Integer/Tuple_) – The linear index to convert back to coordinates.
- **shape** (_Shape_) – Shape of the layout defining the size of each mode
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions.

**Returns:**
: The result of applying the layout transformation to the provided coordinate.

**Return type:**
: Coord

**Examples:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    coord = cute.idx2crd(11, (5, 4))
    # idx2crd is always col-major
    # For shape (m, n, l, ...), coord = (idx % m, idx // m % n, idx // m // n % l, ...
    # Computed as: (11 % 5, 11 // 5 % 4) = (1, 2)
    print(coord)

foo()  # Expected output: (1, 2)
```

```
`cutlass.cute.``recast_layout`(
```

Recast a layout from one data type to another.

**Parameters:**
: - **new_type_bits** (_int_) – The new data type bits
- **old_type_bits** (_int_) – The old data type bits
- **src_layout** (_Union__[__Layout__,__ComposedLayout__]_) – The layout to recast
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions.

**Returns:**
: The recast layout

**Return type:**
: Layout or ComposedLayout

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout
    L = cute.make_layout((2, 3, 4))
    # Recast the layout to a different data type
    L_recast = cute.recast_layout(16, 8, L)
    print(L_recast)
foo()  # Expected output: (2, 3, 4)
```

```
`cutlass.cute.``slice_and_offset`(_`coord`_, _`src`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.slice_and_offset "Link to this definition")
```

```
`cutlass.cute.``recast_ptr`(
```

```
`cutlass.cute.``make_ptr`(
```

```
`cutlass.cute.``composition`(
```

Compose two layout representations using the CuTe layout algebra.

Compose a left-hand layout (or tensor) with a right-hand operand into a new layout R, such that
for every coordinate c in the domain of the right-hand operand, the composed layout satisfies:

> R(c) = A(B(c))

where A is the left-hand operand provided as `lhs` and B is the right-hand operand provided as
`rhs`. In this formulation, B defines the coordinate domain while A applies its transformation to
B’s output, and the resulting layout R inherits the stride and shape adjustments from A.

**Satisfies:**

  cute.shape(cute.composition(lhs, rhs)) is compatible with cute.shape(rhs)

**Parameters:**
: - **lhs** (_Layout__or__Tensor_) – The left-hand operand representing the transformation to be applied.
- **rhs** (_Layout__,__Shape__, or__Tile__, or__int__or__tuple_) – The right-hand operand defining the coordinate domain. If provided as an int or tuple,
it will be converted to a tile layout.
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions.

**Returns:**
: A new composed layout R, such that for all coordinates c in the domain of `rhs`,
R(c) = lhs(rhs(c)).

**Return type:**
: Layout or Tensor

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout that maps (i,j) to i*4 + j
    L1 = cute.make_layout((2, 3), stride=(4, 1))
    # Create a layout that maps (i,j) to i*3 + j
    L2 = cute.make_layout((3, 4), stride=(3, 1))
    # Compose L1 and L2
    L3 = cute.composition(L1, L2)
    # L3 now maps coordinates through L2 then L1
```

```
`cutlass.cute.``complement`(
```

Compute the complement layout of the input layout with respect to the cotarget.

The complement of a layout A with respect to cotarget n is a layout A* such that
for every k in Z_n and c in the domain of A, there exists a unique c* in the domain
of A* where k = A(c) + A*(c*).

This operation is useful for creating layouts that partition a space in complementary ways,
such as row and column layouts that together cover a matrix.

**Parameters:**
: - **input** (_Layout_) – The layout to compute the complement of
- **cotarget** (_Union__[__Layout__,__Shape__]_) – The target layout or shape that defines the codomain
- **loc** (_optional_) – Optional location information for IR diagnostics
- **ip** (_optional_) – Optional instruction pointer or context for underlying IR functions

**Returns:**
: The complement layout

**Return type:**
: Layout

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a right-major layout for a 4x4 matrix
    row_layout = cute.make_layout((4, 4), stride=(4, 1))
    # Create a left-major layout that complements the row layout
    col_layout = cute.complement(row_layout, 16)
    # The two layouts are complementary under 16
```

```
`cutlass.cute.``right_inverse`(
```

```
`cutlass.cute.``left_inverse`(
```

```
`cutlass.cute.``logical_product`(
```

```
`cutlass.cute.``zipped_product`(
```

```
`cutlass.cute.``tiled_product`(
```

```
`cutlass.cute.``flat_product`(
```

```
`cutlass.cute.``raked_product`(
```

```
`cutlass.cute.``blocked_product`(
```

```
`cutlass.cute.``logical_divide`(
```

```
`cutlass.cute.``zipped_divide`(
```

`zipped_divide` is `logical_divide` with Tiler modes and Rest modes gathered together: `(Tiler,Rest)`

- When Tiler is Layout, this has no effect as `logical_divide` results in the same.
- When Tiler is `Tile` (nested tuple of `Layout`) or `Shape`, this zips modes into standard form
`((BLK_A,BLK_B),(a,b,x,y))`

For example, if `target` has shape `(s, t, r)` and `tiler` has shape `(BLK_A, BLK_B)`,
then the result will have shape `((BLK_A, BLK_B), (ceil_div(s, BLK_A), ceil_div(t, BLK_B), r))`.

**Parameters:**
: - **target** (_Layout__or__Tensor_) – The layout or tensor to partition.
- **tiler** (_Tiler_) – The tiling specification (can be a Layout, Shape, Tile).
- **loc** (_optional_) – Optional MLIR IR location information.
- **ip** (_optional_) – Optional MLIR IR insertion point.

**Returns:**
: A zipped (partitioned) version of the target.

**Return type:**
: Layout or Tensor

**Example:**

```python
layout = cute.make_layout((128, 64), stride=(64, 1))
tiler = (8, 8)
result = cute.zipped_divide(layout, tiler)  # result shape: ((8, 8), (16, 8))
```

```
`cutlass.cute.``tiled_divide`(
```

```
`cutlass.cute.``flat_divide`(
```

```
`cutlass.cute.``max_common_layout`(
```

```
`cutlass.cute.``max_common_vector`(
```

```
`cutlass.cute.``tile_to_shape`(
```

```
`cutlass.cute.``local_partition`(
```

```
`cutlass.cute.``local_tile`(
```

Partition a tensor into tiles using a tiler and extract a single tile at the provided coordinate.

The `local_tile` operation applies a `zipped_divide` to split the `input` tensor by the `tiler`
and then slices out a single tile using the provided *coord*. This is commonly used for extracting block-,
thread-, or CTA-level tiles for parallel operations.

$$
\[\text{local_tile}(input, tiler, coord) = \text{zipped_divide}(input, tiler)[coord]\]
$$

This function corresponds to the CUTE/C++ *local_tile* utility:
<https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#local-tile>

**Parameters:**
: - **input** (_Tensor_) – The input tensor to partition into tiles.
- **tiler** (_Tiler_) – The tiling specification (can be a Layout, Shape, Tile).
- **coord** (_Coord_) – The coordinate to select within the remainder (“rest”) modes after tiling.
This selects which tile to extract.
- **proj** (_XTuple__,__optional_) – (Optional) Projection onto tiling modes; specify to project out unused tiler modes,
e.g., when working with projections of tilers in multi-mode partitioning.
Default is None for no projection.
- **loc** (_Any__,__optional_) – (Optional) MLIR location, for diagnostic/debugging.
- **ip** (_Any__,__optional_) – (Optional) MLIR insertion point, used in IR building context.

**Returns:**
: A new tensor representing the local tile selected at the given coordinate.

**Return type:**
: Tensor

**Examples**

1. Tiling a 2D tensor and extracting a tile:

> ```python
> # input: (16, 24)
> tensor : cute.Tensor
> tiler = (2, 4)
> coord = (1, 1)
>
> # output: (8, 6)
> # - zipped_divide(tensor, tiler)     -> ((2, 4), (8, 6))
> # - local_tile(tensor, tiler, coord) -> (8, 6)
> result = cute.local_tile(tensor, tiler=tiler, coord=coord)
> ```
2. Using a stride projection for specialized tiling:

> ```python
> # input: (16, 24)
> tensor : cute.Tensor
> tiler = (2, 2, 4)
> coord = (0, 1, 1)
> proj = (1, None, 1)
>
> # output: (8, 6)
> # projected_tiler: (2, 4)
> # projected_coord: (0, 1)
> # - zipped_divide(tensor, projected_tiler)               -> ((2, 4), (8, 6))
> # - local_tile(tensor, projected_tiler, projected_coord) -> (8, 6)
> result = cute.local_tile(tensor, tiler=tiler, coord=coord, proj=proj)
> ```

```
`cutlass.cute.``make_layout_image_mask`(
```

Makes a 16-bit integer mask of the image of a layout sliced at a given mode
and accounting for the offset given by the input coordinate for the other modes.

```
`cutlass.cute.``leading_dim`(
```

Find the leading dimension of a shape and stride.

**Parameters:**
: - **shape** (_Shape_) – The shape of the tensor or layout
- **stride** (_Stride_) – The stride of the tensor or layout

**Returns:**
: The leading dimension index or indices

**Return type:**
: Union[int, Tuple[int, …], None]

The return value depends on the stride pattern:

> - If a single leading dimension is found, returns an integer index
> - If nested leading dimensions are found, returns a tuple of indices
> - If no leading dimension is found, returns None

```
`cutlass.cute.``make_layout_tv`(
```

Create a thread-value layout by repeating the val_layout over the thr_layout.

This function creates a thread-value layout that maps between `(thread_idx, value_idx)`
coordinates and logical `(M,N)` coordinates. The thread and value layouts must be compact to ensure
proper partitioning.

This implements the thread-value partitioning pattern where data is partitioned
across threads and values within each thread.

**Parameters:**
: - **thr_layout** (_Layout_) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
- **val_layout** (_Layout_) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs within each thread
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tuple containing `tiler_mn` and `layout_tv`

**Return type:**
: Tuple[Shape, Layout]

**where:**

  - `tiler_mn` is tiler and `shape(tiler_mn)` is compatible with `shape(zipped_divide(x, tiler_mn))[0]`
  - `layout_tv`: Thread-value layout mapping (thread_idx, value_idx) -> (M,N)

**Example:**

**The below code creates a TV Layout that maps thread/value coordinates to the logical coordinates in a `(4,6)` tensor:**

  - _Tiler MN_: `(4,6)`
  - _TV Layout_: `((3,2),(2,2)):((8,2),(4,1))`

```python
thr_layout = cute.make_layout((2, 3), stride=(3, 1))
val_layout = cute.make_layout((2, 2), stride=(2, 1))
tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
```

|  | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | T0, V0 | T0, V1 | T1, V0 | T1, V1 | T2, V0 | T2, V1 |
| 1 | T0, V2 | T0, V3 | T1, V2 | T1, V3 | T2, V2 | T2, V3 |
| 2 | T3, V0 | T3, V1 | T4, V0 | T4, V1 | T5, V0 | T5, V1 |
| 3 | T3, V2 | T3, V3 | T4, V2 | T4, V3 | T5, V2 | T5, V3 |

```
`cutlass.cute.``get_nonswizzle_portion`(
```

Extract the non-swizzle portion from a layout.

For a simple Layout, the entire layout is considered non-swizzled and is returned as-is.
For a ComposedLayout, the inner layout (non-swizzled portion) is extracted and returned,
effectively separating the base layout from any swizzle transformation that may be applied.

**Parameters:**
: - **layout** (_Union__[__Layout__,__ComposedLayout__]_) – A Layout or ComposedLayout from which to extract the non-swizzle portion.
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional

**Returns:**
: The non-swizzle portion of the input layout. For Layout objects, returns the layout itself.
For ComposedLayout objects, returns the outer layout component.

**Return type:**
: Layout

**Raises:**
: **TypeError** – If the layout is neither a Layout nor a ComposedLayout.

```
`cutlass.cute.``get_swizzle_portion`(
```

Extract or create the swizzle portion from a layout.

For a simple Layout (which has no explicit swizzle), a default identity swizzle is created.
For a ComposedLayout, the outer layout is checked and returned if it is a Swizzle object.
Otherwise, a default identity swizzle is created. The default identity swizzle has parameters
(0, 4, 3), which represents a no-op swizzle transformation.

**Parameters:**
: - **layout** (_Union__[__Layout__,__ComposedLayout__]_) – A Layout or ComposedLayout from which to extract the swizzle portion.
- **loc** (_optional_) – Optional location information for IR diagnostics.
- **ip** (_optional_) – Optional

**Returns:**
: The swizzle portion of the layout. For Layout objects or ComposedLayout objects without
a Swizzle outer component, returns a default identity swizzle (0, 4, 3). For ComposedLayout
objects with a Swizzle outer component, returns that swizzle.

**Return type:**
: [Swizzle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Swizzle "cutlass.cute.Swizzle")

**Raises:**
: **TypeError** – If the layout is neither a Layout nor a ComposedLayout.

```
`cutlass.cute.``transform_leaf`(_`f`_, _`*``args`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.transform_leaf "Link to this definition")
```

Apply a function to the leaf nodes of nested tuple structures.

This function traverses nested tuple structures in parallel and applies the function f
to corresponding leaf nodes. All input tuples must have the same nested structure.

**Parameters:**
: - **f** (_Callable_) – Function to apply to leaf nodes
- **args** – One or more nested tuple structures with matching profiles

**Returns:**
: A new nested tuple with the same structure as the inputs, but with leaf values transformed by f

**Raises:**
: **TypeError** – If the input tuples have different nested structures

**Example:**

```python
>>> transform_leaf(lambda x: x + 1, (1, 2))
(2, 3)
>>> transform_leaf(lambda x, y: x + y, (1, 2), (3, 4))
(4, 6)
>>> transform_leaf(lambda x: x * 2, ((1, 2), (3, 4)))
((2, 4), (6, 8))
```

```
`cutlass.cute.``find_if`(
```

```
`cutlass.cute.``find`(
```

Find the first position of a value `x` in a hierarchical structure `t`.

Searches for the first occurrence of x in t, optionally excluding positions
where a comparison value matches. The search can traverse nested structures
and returns either a single index or a tuple of indices for nested positions.

**Parameters:**
: - **t** (_Union__[__tuple__,__ir.Value__,__int__]_) – The search space
- **x** (_int_) – The static integer x to search for

**Returns:**
: Index if found at top level, tuple of indices showing nested position, or None if not found

**Return type:**
: Union[int, Tuple[int, …], None]

```
`cutlass.cute.``flatten_to_tuple`(
```

Flattens a potentially nested tuple structure into a flat tuple.

This function recursively traverses the input structure and flattens it into
a single-level tuple, preserving the order of elements.

**Parameters:**
: **a** (_Union__[__IntTuple__,__Coord__,__Shape__,__Stride__]_) – The structure to flatten

**Returns:**
: A flattened tuple containing all elements from the input

**Return type:**
: tuple

**Examples:**

```python
flatten_to_tuple((1, 2, 3))       # Returns (1, 2, 3)
flatten_to_tuple(((1, 2), 3))     # Returns (1, 2, 3)
flatten_to_tuple((1, (2, (3,))))  # Returns (1, 2, 3)
```

```
`cutlass.cute.``unflatten`(
```

Unflatten a flat tuple into a nested tuple structure according to a profile.

This function transforms a flat sequence of elements into a nested tuple structure
that matches the structure defined by the profile parameter. It traverses the profile
structure and populates it with elements from the sequence.

sequence must be long enough to fill the profile. Raises RuntimeError if it is not.

**Parameters:**
: - **sequence** (_Union__[__Tuple__[__Any__,__...__]__,__List__[__Any__]__,__Iterable__[__Any__]__]_) – A flat sequence of elements to be restructured
- **profile** (_XTuple_) – A nested tuple structure that defines the shape of the output

**Returns:**
: A nested tuple with the same structure as profile but containing elements from sequence

**Return type:**
: XTuple

**Examples:**

```python
unflatten([1, 2, 3, 4], ((0, 0), (0, 0)))  # Returns ((1, 2), (3, 4))
```

```
`cutlass.cute.``product`(
```

```
`cutlass.cute.``product_like`(
```

Return product of the given IntTuple or Shape at leaves of *target_profile*.

This function computes products according to the structure defined by target_profile.

**Parameters:**
: - **a** (_IntTuple__or__Shape_) – The input tuple or shape
- **target_profile** (_XTuple_) – The profile that guides how products are computed
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The resulting tuple with products computed according to target_profile

**Return type:**
: IntTuple or Shape

**Raises:**
: - **TypeError** – If inputs have incompatible types
- **ValueError** – If inputs have incompatible shapes

```
`cutlass.cute.``product_each`(
```

```
`cutlass.cute.``elem_less`(
```

```
`cutlass.cute.``tuple_cat`(_`*``tuples`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.tuple_cat "Link to this definition")
```

Concatenate multiple tuples into a single tuple.

This function takes any number of tuples and concatenates them into a single tuple.
Non-tuple arguments are treated as single-element tuples.

**Parameters:**
: **tuples** (_tuple__or__any_) – Variable number of tuples to concatenate

**Returns:**
: A single concatenated tuple

**Return type:**
: tuple

**Examples:**

```python
>>> tuple_cat((1, 2), (3, 4))
(1, 2, 3, 4)
>>> tuple_cat((1,), (2, 3), (4,))
(1, 2, 3, 4)
>>> tuple_cat(1, (2, 3))
(1, 2, 3)
```

```
`cutlass.cute.``transform_apply`(_`*``args`_, _`f``:` `Callable`_, _`g``:` `Callable`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.transform_apply "Link to this definition")
```

Transform elements of tuple(s) with f, then apply g to all results.

This function applies f to corresponding elements across input tuple(s),
then applies g to all transformed results. It mimics the C++ CuTe implementation.

Supports multiple signatures:
- transform_apply(t, f, g): For single tuple, computes g(f(t[0]), f(t[1]), …)
- transform_apply(t0, t1, f, g): For two tuples, computes g(f(t0[0], t1[0]), f(t0[1], t1[1]), …)
- transform_apply(t0, t1, t2, …, f, g): For multiple tuples of same length

For non-tuple inputs, f is applied to the input(s) and g is applied to that single result.

**Parameters:**
: - **args** – One or more tuples (or non-tuples) to transform
- **f** (_Callable_) – The function to apply to each element (or corresponding elements across tuples)
- **g** (_Callable_) – The function to apply to all transformed elements
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: The result of applying g to all transformed elements

**Return type:**
: any

**Examples:**

```python
>>> transform_apply((1, 2, 3), f=lambda x: x * 2, g=lambda *args: sum(args))
12  # (1*2 + 2*2 + 3*2) = 12
>>> transform_apply((1, 2), f=lambda x: (x, x+1), g=tuple_cat)
(1, 2, 2, 3)
>>> transform_apply((1, 2), (3, 4), f=lambda x, y: x + y, g=lambda *args: args)
(4, 6)
```

```
`cutlass.cute.``filter_tuple`(_`*``args`_, _`f``:` `Callable`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.filter_tuple "Link to this definition")
```

Filter and flatten tuple elements by applying a function.

The function f should return tuples, which are then concatenated together
to produce the final result. This is useful for filtering and transforming
tuple structures in a single pass.

**Parameters:**
: - **t** (_Union__[__tuple__,__ir.Value__,__int__]_) – The tuple to filter
- **f** (_Callable_) – The function to apply to each element of t
- **loc** (_optional_) – Source location for MLIR, defaults to None
- **ip** (_optional_) – Insertion point, defaults to None

**Returns:**
: A concatenated tuple of all results

**Return type:**
: tuple

**Examples:**

```python
>>> # Keep only even numbers, wrapped in tuples
>>> filter_tuple((1, 2, 3, 4), lambda x: (x,) if x % 2 == 0 else ())
(2, 4)
>>> # Duplicate each element
>>> filter_tuple((1, 2, 3), lambda x: (x, x))
(1, 1, 2, 2, 3, 3)
```

```
_`class`_`cutlass.cute.``TensorSSA`(_`*``args``:` `Any`_, _`**``kwargs``:` `Any`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "Link to this definition")
```

Bases: `ArithValue`

A class representing thread local data from CuTe Tensor in value semantic and immutable.

**Parameters:**
: - **value** (_ir.Value_) – Flatten vector as ir.Value holding logic data of SSA Tensor
- **shape** (_Shape_) – The nested shape in CuTe of the vector
- **dtype** (_Type__[__Numeric__]_) – Data type of the tensor elements

**Variables:**
: - **_shape** – The nested shape in CuTe of the vector
- **_dtype** – Data type of the tensor elements

**Raises:**
: **ValueError** – If shape is not static

```
`__init__`(
```

Initialize a new TensorSSA object.

**Parameters:**
: - **value** (_ir.Value_) – Flatten vector as ir.Value holding logic data of SSA Tensor
- **shape** (_Shape_) – The nested shape in CuTe of the vector
- **dtype** (_Type__[__Numeric__]_) – Data type of the tensor elements

**Raises:**
: **ValueError** – If shape is not static

```
_`property`_`dtype`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA.dtype "Link to this definition")
```

```
_`property`_`element_type`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA.element_type "Link to this definition")
```

```
_`property`_`shape`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA.shape "Link to this definition")
```

```
`_apply_op`(
```

```
`_apply_op`(
```

```
`_apply_op`(
```

```
`apply_op`(
```

Apply a binary operation to this tensor and another operand.

This is a public interface to the internal _apply_op method, providing
a stable API for external users who need to apply custom operations.

**Parameters:**
: - **op** – The operation function (e.g., operator.add, operator.mul, etc.)
- **other** – The other operand (TensorSSA, ArithValue, or scalar)
- **flip** – Whether to flip the operands (for right-hand operations)
- **loc** – MLIR location (optional)
- **ip** – MLIR insertion point (optional)

**Returns:**
: The result of the operation

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

**Example**

```default
>>> tensor1 = cute.Tensor(...)
>>> tensor2 = cute.Tensor(...)
>>> result = tensor1.apply_op(operator.add, tensor2)
>>> # Equivalent to: tensor1 + tensor2
```

```
`broadcast_to`(
```

Broadcast the tensor to the target shape.

```
`_flatten_shape_and_coord`(_`crd`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA._flatten_shape_and_coord "Link to this definition")
```

```
`_build_result`(
```

```
`reshape`(
```

Reshape the tensor to a new shape.

**Parameters:**
: **shape** (_Shape_) – The new shape to reshape to.

**Returns:**
: A new tensor with the same elements but with the new shape.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

**Raises:**
: - **NotImplementedError** – If dynamic size is not supported
- **ValueError** – If the new shape is not compatible with the current shape

```
`to`(
```

Convert the tensor to a different numeric type.

**Parameters:**
: **dtype** (_Type__[__Numeric__]_) – The target numeric type to cast to.

**Returns:**
: A new tensor with the same shape but with elements cast to the target type.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

**Raises:**
: - **TypeError** – If dtype is not a subclass of Numeric.
- **NotImplementedError** – If dtype is an unsigned integer type.

```
`ir_value`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA.ir_value "Link to this definition")
```

```
`ir_value_int8`(_`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA.ir_value_int8 "Link to this definition")
```

Returns int8 ir value of Boolean tensor.
When we need to store Boolean tensor ssa, use ir_value_int8().

**Parameters:**
: - **loc** (_Optional__[__Location__]__,__optional_) – Source location information, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point for MLIR operations, defaults to None

**Returns:**
: The int8 value of this Boolean

**Return type:**
: ir.Value

```
`reduce`(
```

Perform reduce on selected modes with given predefined reduction op.

**Parameters:**
: - **op** (_operator_) – The reduction operator to use (operator.add or operator.mul)
- **init_val** (_numeric_) – The initial value for the reduction
- **reduction_profile** (_Coord_) – Specifies which dimensions to reduce. Dimensions marked with *None* are kept.

**Returns:**
: The reduced tensor

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

**Examples:**

```python
reduce(f32 o (4,))
  => f32

reduce(f32 o (4, 5))
  => f32
reduce(f32 o (4, (5, 4)), reduction_profile=(None, 1))
  => f32 o (4,)
reduce(f32 o (4, (5, 4)), reduction_profile=(None, (None, 1)))
  => f32 o (4, (5,))
```

```
`cutlass.cute.``make_tensor`(
```

Creates a tensor by composing an engine (iterator/pointer) with a layout.

A tensor is defined as T = E ∘ L, where E is an engine (array, pointer, or counting iterator)
and L is a layout that maps logical coordinates to physical offsets. The tensor
evaluates coordinates by applying the layout mapping and dereferencing the engine
at the resulting offset.

**Parameters:**
: - **iterator** (_Union__[__Pointer__,__IntTuple__,__ir.Value__]_) – Engine component that provides data access capabilities. Can be:
- A pointer (Pointer type)
- An integer or integer tuple for coordinate tensors
- A shared memory descriptor (SmemDescType)
- **layout** (_Union__[__Shape__,__Layout__,__ComposedLayout__]_) – Layout component that defines the mapping from logical coordinates to
physical offsets. Can be:
- A shape tuple that will be converted to a layout
- A Layout object
- A ComposedLayout object (must be a normal layout)
- **loc** (_Optional__[__Location__]_) – Source location for MLIR operation tracking, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for MLIR operation, defaults to None

**Returns:**
: A tensor object representing the composition E ∘ L

**Return type:**
: Tensor

**Raises:**
: - **TypeError** – If iterator type is not a supported type
- **ValueError** – If layout is a composed layout with customized inner functions

**Examples:**

```python
# Create a tensor with row-major layout from a pointer
ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
layout = make_layout((64, 128), stride=(128, 1))
tensor = make_tensor(ptr, layout)

# Create a tensor with hierarchical layout in shared memory
smem_ptr = make_ptr(Float16, base_ptr, AddressSpace.smem)
layout = make_layout(((128, 8), (1, 4, 1)), stride=((32, 1), (0, 8, 4096)))
tensor = make_tensor(smem_ptr, layout)

# Create a coordinate tensor
layout = make_layout(2, stride=16 * E(0))
tensor = make_tensor(5, layout)  # coordinate tensor with iterator starting at 5
```

**Notes**

- The engine (iterator) must support random access operations
- Common engine types include raw pointers, arrays, and random-access iterators
- The layout defines both the shape (logical dimensions) and stride (physical mapping)
- Supports both direct coordinate evaluation T(c) and partial evaluation (slicing)
- ComposedLayouts must be “normal” layouts (no inner functions)
- For coordinate tensors, the iterator is converted to a counting sequence

```
`cutlass.cute.``make_identity_tensor`(
```

Creates an identity tensor with the given shape.

An identity tensor maps each coordinate to itself, effectively creating a counting
sequence within the shape’s bounds. This is useful for generating coordinate indices
or creating reference tensors for layout transformations.

**Parameters:**
: - **shape** (_Shape_) – The shape defining the tensor’s dimensions. Can be a simple integer
sequence or a hierarchical structure ((m,n),(p,q))
- **loc** (_Optional__[__Location__]_) – Source location for MLIR operation tracking, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for MLIR operation, defaults to None

**Returns:**
: A tensor that maps each coordinate to itself

**Return type:**
: Tensor

**Examples:**

```python
# Create a simple 1D coord tensor
tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

# Create a 2D coord tensor
tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Create hierarchical coord tensor
tensor = make_identity_tensor(((2,1),3))
# [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
```

**Notes**

- The shape parameter follows CuTe’s IntTuple concept
- Coordinates are ordered colexicographically
- Useful for generating reference coordinates in layout transformations

```
`cutlass.cute.``make_fragment`(
```

```
`cutlass.cute.``make_fragment_like`(_`src`_, _`dtype``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_fragment_like "Link to this definition")
```

```
`cutlass.cute.``make_rmem_tensor_like`(
```

**Creates a tensor in register memory with the same shape as the input layout but**

  compact col-major strides. This is equivalent to calling *make_rmem_tensor(make_layout_like(tensor))*.

This function allocates a tensor in register memory (rmem) usually on stack with
with the compact layout like the source. The tensor will have elements of the
specified numeric data type or the same as the source.

**Parameters:**
: - **src** (_Union__[__Layout__,__ComposedLayout__,__Tensor__]_) – The source layout or tensor whose shape will be matched
- **dtype** (_Type__[__Numeric__]__,__optional_) – The element type for the fragment tensor, defaults to None
- **loc** (_Location__,__optional_) – Source location for MLIR operations, defaults to None
- **ip** (_InsertionPoint__,__optional_) – Insertion point for MLIR operations, defaults to None

**Returns:**
: A new layout or fragment tensor with matching shape

**Return type:**
: Union[Layout, Tensor]

**Examples:**

Creating a rmem tensor from a tensor:

```python
smem_tensor = cute.make_tensor(smem_ptr, layout)
rmem_tensor = cute.make_rmem_tensor_like(smem_tensor, cutlass.Float32)
# frag_tensor will be a register-backed tensor with the same shape
```

Creating a fragment with a different element type:

```python
tensor = cute.make_tensor(gmem_ptr, layout)
rmem_bool_tensor = cute.make_rmem_tensor_like(tensor, cutlass.Boolean)
# bool_frag will be a register-backed tensor with Boolean elements
```

**Notes**

- When used with a Tensor, if a type is provided, it will create a new
fragment tensor with that element type.
- For layouts with ScaledBasis strides, the function creates a fragment
from the shape only.
- This function is commonly used in GEMM and other tensor operations to
create register storage for intermediate results.

```
`cutlass.cute.``make_rmem_tensor`(
```

Creates a tensor in register memory with the specified layout/shape and data type.

This function allocates a tensor in register memory (rmem) usually on stack with
either a provided layout or creates a new layout from the given shape. The tensor
will have elements of the specified numeric data type.

**Parameters:**
: - **layout_or_shape** (_Union__[__Layout__,__Shape__]_) – Either a Layout object defining the tensor’s memory organization,
or a Shape defining its dimensions
- **dtype** (_Type__[__Numeric__]_) – The data type for tensor elements (must be a Numeric type)
- **loc** (_Optional__[__Location__]_) – Source location for MLIR operation tracking, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for MLIR operation, defaults to None

**Returns:**
: A tensor allocated in register memory

**Return type:**
: Tensor

**Examples:**

```python
# Create rmem tensor with explicit layout
layout = make_layout((128, 32))
tensor = make_rmem_tensor(layout, cutlass.Float16)

# Create rmem tensor directly from shape
tensor = make_rmem_tensor((64, 64), cutlass.Float32)
```

**Notes**

- Uses 32-byte alignment to support .128 load/store operations
- Boolean types are stored as 8-bit integers
- Handles both direct shapes and Layout objects

```
`cutlass.cute.``recast_tensor`(
```

Recast a tensor to a different data type by changing the element interpretation.

This function reinterprets the memory of a tensor with a different element type,
adjusting both the iterator pointer type and the layout to maintain consistency.

**Parameters:**
: - **src** (_Tensor_) – The source tensor to recast
- **dtype** (_Type__[__Numeric__]_) – The target data type for tensor elements
- **swizzle** (_Optional__,__unused_) – Optional swizzle parameter (reserved for future use), defaults to None
- **loc** (_Optional__[__Location__]_) – Source location for MLIR operation tracking, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for MLIR operation, defaults to None

**Returns:**
: A new tensor with the same memory but reinterpreted as dtype

**Return type:**
: Tensor

**Raises:**
: **TypeError** – If dtype is not a subclass of Numeric

**Examples:**

```python
# Create a Float32 tensor
tensor_f32 = make_rmem_tensor((4, 8), Float32)

# Recast to Int32 to manipulate bits
tensor_i32 = recast_tensor(tensor_f32, Int32)

# Both tensors share the same memory, but interpret it differently
```

```
`cutlass.cute.``domain_offset`(
```

Offset the tensor domain by the given coordinate.

This function creates a new tensor by offsetting the iterator/pointer of the input tensor
by the amount corresponding to the given coordinate in its layout.

**Parameters:**
: - **coord** (_Coord_) – The coordinate offset to apply
- **tensor** (_Tensor_) – The source tensor to offset
- **loc** (_Optional__[__Location__]_) – Source location for MLIR operation tracking, defaults to None
- **ip** (_Optional__[__InsertionPoint__]_) – Insertion point for MLIR operation, defaults to None

**Returns:**
: A new tensor with the offset iterator

**Return type:**
: Tensor

**Raises:**
: **ValueError** – If the tensor type doesn’t support domain offsetting

**Examples:**

```python
# Create a tensor with a row-major layout
ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
layout = make_layout((64, 128), stride=(128, 1))
tensor = make_tensor(ptr, layout)

# Offset by coordinate (3, 5)
offset_tensor = domain_offset((3, 5), tensor)
# offset_tensor now points to element at (3, 5)
```

```
`cutlass.cute.``print_tensor`(
```

Print content of the tensor in human readable format.

Outputs the tensor data in a structured format showing both metadata
and the actual data values. The output includes tensor type information,
layout details, and a formatted array representation of the values.

**Parameters:**
: - **tensor** (_Tensor_) – The tensor to print
- **verbose** (_bool_) – If True, includes additional debug information in the output
- **loc** (_source location__,__optional_) – Source location where it’s called, defaults to None
- **ip** (_insertion pointer__,__optional_) – Insertion pointer for IR generation, defaults to None

**Raises:**
: **NotImplementedError** – If the tensor type doesn’t support trivial dereferencing

**Example output:**

```text
tensor(raw_ptr<@..., Float32, generic, align(4)> o (8,5):(5,1), data=
       [[-0.4326, -0.5434,  0.1238,  0.7132,  0.8042],
        [-0.8462,  0.9871,  0.4389,  0.7298,  0.6948],
        [ 0.3426,  0.5856,  0.1541,  0.2923,  0.6976],
        [-0.1649,  0.8811,  0.1788,  0.1404,  0.2568],
        [-0.2944,  0.8593,  0.4171,  0.8998,  0.1766],
        [ 0.8814,  0.7919,  0.7390,  0.4566,  0.1576],
        [ 0.9159,  0.7577,  0.6918,  0.0754,  0.0591],
        [ 0.6551,  0.1626,  0.1189,  0.0292,  0.8655]])
```

```
`cutlass.cute.``full`(
```

Return a new TensorSSA of given shape and type, filled with fill_value.

**Parameters:**
: - **shape** (_tuple_) – Shape of the new tensor.
- **fill_value** (_scalar_) – Value to fill the tensor with.
- **dtype** (_Type__[__Numeric__]_) – Data type of the tensor.

**Returns:**
: Tensor of fill_value with the specified shape and dtype.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``full_like`(
```

Return a full TensorSSA with the same shape and type as a given array.

**Parameters:**
: - **a** (_array_like_) – The shape and data-type of *a* define these same attributes of the returned array.
- **fill_value** (_array_like_) – Fill value.
- **dtype** (_Union__[__None__,__Type__[__Numeric__]__]__,__optional_) – Overrides the data type of the result, defaults to None

**Returns:**
: Tensor of *fill_value* with the same shape and type as *a*.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

> **See also**
>
> [`empty_like()`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.empty_like "cutlass.cute.empty_like"): Return an empty array with shape and type of input.
> [`ones_like()`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ones_like "cutlass.cute.ones_like"): Return an array of ones with shape and type of input.
> [`zeros_like()`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.zeros_like "cutlass.cute.zeros_like"): Return an array of zeros with shape and type of input.
> [`full()`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.full "cutlass.cute.full"): Return a new array of given shape filled with value.

**Examples:**

```python
frg = cute.make_rmem_tensor((2, 3), Float32)
a = frg.load()
b = cute.full_like(a, 1.0)
```

```
`cutlass.cute.``empty_like`(_`a`_, _`dtype``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.empty_like "Link to this definition")
```

Return a new TensorSSA with the same shape and type as a given array, without initializing entries.

**Parameters:**
: - **a** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – The shape and data-type of *a* define these same attributes of the returned array.
- **dtype** (_Type__[__Numeric__]__,__optional_) – Overrides the data type of the result, defaults to None

**Returns:**
: Uninitialized tensor with the same shape and type (unless overridden) as *a*.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``ones_like`(_`a`_, _`dtype``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ones_like "Link to this definition")
```

Return a TensorSSA of ones with the same shape and type as a given array.

**Parameters:**
: - **a** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – The shape and data-type of *a* define these same attributes of the returned array.
- **dtype** (_Type__[__Numeric__]__,__optional_) – Overrides the data type of the result, defaults to None

**Returns:**
: Tensor of ones with the same shape and type (unless overridden) as *a*.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``zeros_like`(_`a`_, _`dtype``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.zeros_like "Link to this definition")
```

Return a TensorSSA of zeros with the same shape and type as a given array.

**Parameters:**
: - **a** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – The shape and data-type of *a* define these same attributes of the returned array.
- **dtype** (_Type__[__Numeric__]__,__optional_) – Overrides the data type of the result, defaults to None

**Returns:**
: Tensor of zeros with the same shape and type (unless overridden) as *a*.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``where`(
```

Return elements chosen from x or y depending on condition; will auto broadcast x or y if needed.

**Parameters:**
: - **cond** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – Where True, yield x, where False, yield y.
- **x** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Values from which to choose when condition is True.
- **y** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Values from which to choose when condition is False.

**Returns:**
: A tensor with elements from x where condition is True, and elements from y where condition is False.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``any_`(
```

Test whether any tensor element evaluates to True.

**Parameters:**
: **x** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – Input tensor.

**Returns:**
: Returns a TensorSSA scalar containing True if any element of x is True, False otherwise.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
`cutlass.cute.``all_`(
```

Test whether all tensor elements evaluate to True.

**Parameters:**
: **x** ([_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")) – Input tensor.

**Returns:**
: Returns a TensorSSA scalar containing True if all elements of x are True, False otherwise.

**Return type:**
: [TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")

```
_`class`_`cutlass.cute.``Atom`(_`op``:` `Op`_, _`trait``:` `Trait`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom "Link to this definition")
```

Bases: `ABC`

Atom base class.

An Atom is the composition of

- a MMA or Copy Operation;
- an internal MMA or Copy Trait.

An Operation is a pure Python class that is used to model a specific MMA or Copy instruction.
The Trait wraps the underlying IR Value and provides access to the metadata of the instruction
encoded using CuTe Layouts. When the Trait can be constructed straighforwardly from an
Operation, the `make_mma_atom` or `make_copy_atom` API should be used. There are cases where
constructing the metadata is not trivial and requires more information, for example to determine
the number of bytes copied per TMA instruction (“the TMA vector length”). In such cases,
dedicated helper functions are provided with an appropriate API such that the Atom is
constructed internally in an optimal fashion for the user.

```
`__init__`(
```

```
_`property`_`op`_`:` `Op`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom.op "Link to this definition")
```

```
_`property`_`type`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom.type "Link to this definition")
```

```
`set`(_`modifier`_, _`value`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `None`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom.set "Link to this definition")
```

Sets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
```

The `set` method provides a way to the user to modify such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

```
`get`(_`field`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_) → `Any`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom.get "Link to this definition")
```

Gets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
accum = tiled_mma.get(cute.nvgpu.tcgen05.Field.ACCUMULATE)
```

The `get` method provides a way to the user to access such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

```
`with_`(_`*`_, _`loc``=``None`_, _`ip``=``None`_, _`**``kwargs`_) → [`Atom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom "cutlass.cute.atom.Atom")[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom.with_ "Link to this definition")
```

Returns a new Atom with the new Operation and Trait with the given runtime state. The runtime state
is provided as keyword arguments and it is Atom-specific.

```python
tiled_copy = cute.make_tiled_copy(tma_copy_op)
new_tiled_copy = tiled_copy.with_(tma_bar_ptr=tma_bar_ptr, cache_policy=cute.CacheEvictionPriority.EVICT_LAST)
```

The `with_` method provides a way to the user to modify such runtime state or create an executable Atom
(e.g. an Executable TMA Load Atom).

```
`_unpack`(_`*`_, _`loc``=``None`_, _`ip``=``None`_, _`**``kwargs`_) → `cutlass._mlir.ir.Value`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom._unpack "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``MmaAtom`(_`op``:` `Op`_, _`trait``:` `Trait`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "Link to this definition")
```

Bases: [`Atom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom "cutlass.cute.atom.Atom")

The MMA Atom class.

```
_`property`_`thr_id`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.thr_id "Link to this definition")
```

```
_`property`_`shape_mnk`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.shape_mnk "Link to this definition")
```

```
_`property`_`tv_layout_A`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.tv_layout_A "Link to this definition")
```

```
_`property`_`tv_layout_B`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.tv_layout_B "Link to this definition")
```

```
_`property`_`tv_layout_C`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.tv_layout_C "Link to this definition")
```

```
`make_fragment_A`(_`input`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.make_fragment_A "Link to this definition")
```

```
`make_fragment_B`(_`input`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.make_fragment_B "Link to this definition")
```

```
`make_fragment_C`(_`input`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom.make_fragment_C "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``CopyAtom`(_`op``:` `Op`_, _`trait``:` `Trait`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "Link to this definition")
```

Bases: [`Atom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.Atom "cutlass.cute.atom.Atom")

The Copy Atom class.

```
_`property`_`value_type`_`:` `Type``[``cutlass.cute.typing.Numeric``]`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom.value_type "Link to this definition")
```

```
_`property`_`thr_id`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom.thr_id "Link to this definition")
```

```
_`property`_`layout_src_tv`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom.layout_src_tv "Link to this definition")
```

```
_`property`_`layout_dst_tv`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom.layout_dst_tv "Link to this definition")
```

```
_`property`_`smem_layout`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom.smem_layout "Link to this definition")
```

Convenience property to access the SMEM layout for TMA copy atoms.

This is a shortcut for `atom.op.smem_layout` that checks if the operation
is a TMA operation and provides a clearer error message if not.

**Returns:**
: The SMEM layout

**Return type:**
: Layout or ComposedLayout

**Raises:**
: - **TypeError** – If the operation is not a TMA operation
- **ValueError** – If the SMEM layout is not set

**Example**

```default
>>> layout = tma_atom.smem_layout  # Instead of tma_atom.op.smem_layout
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``TiledCopy`(_`op``:` `Op`_, _`trait``:` `Trait`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "Link to this definition")
```

Bases: [`CopyAtom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")

The tiled Copy class.

```
_`property`_`layout_tv_tiled`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.layout_tv_tiled "Link to this definition")
```

```
_`property`_`tiler_mn`_`:` `cutlass.cute.typing.Tile`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.tiler_mn "Link to this definition")
```

```
_`property`_`layout_src_tv_tiled`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.layout_src_tv_tiled "Link to this definition")
```

```
_`property`_`layout_dst_tv_tiled`_`:` `cutlass.cute.typing.Layout`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.layout_dst_tv_tiled "Link to this definition")
```

```
_`property`_`size`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.size "Link to this definition")
```

```
`get_slice`(
```

```
`retile`(_`src`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy.retile "Link to this definition")
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``TiledMma`(_`op``:` `Op`_, _`trait``:` `Trait`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "Link to this definition")
```

Bases: [`MmaAtom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "cutlass.cute.atom.MmaAtom")

The tiled MMA class.

```
_`property`_`tv_layout_A_tiled`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.tv_layout_A_tiled "Link to this definition")
```

```
_`property`_`tv_layout_B_tiled`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.tv_layout_B_tiled "Link to this definition")
```

```
_`property`_`tv_layout_C_tiled`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.tv_layout_C_tiled "Link to this definition")
```

```
_`property`_`permutation_mnk`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.permutation_mnk "Link to this definition")
```

```
_`property`_`thr_layout_vmnk`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.thr_layout_vmnk "Link to this definition")
```

```
_`property`_`size`_`:` `int`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.size "Link to this definition")
```

```
`get_tile_size`(_`mode_idx``:` `int`_) → `cutlass.cute.typing.Shape`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.get_tile_size "Link to this definition")
```

```
`get_slice`(
```

```
`_partition_shape`(_`operand_id`_, _`shape`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma._partition_shape "Link to this definition")
```

```
`partition_shape_A`(_`shape_mk`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.partition_shape_A "Link to this definition")
```

```
`partition_shape_B`(_`shape_nk`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.partition_shape_B "Link to this definition")
```

```
`partition_shape_C`(_`shape_mn`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma.partition_shape_C "Link to this definition")
```

```
`_thrfrg`(
```

```
`_thrfrg`(
```

```
`_thrfrg_A`(
```

```
`_thrfrg_B`(
```

```
`_thrfrg_C`(
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``ThrMma`(
```

Bases: [`TiledMma`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")

The thread MMA class for modeling a thread-slice of a tiled MMA.

```
`__init__`(
```

```
_`property`_`thr_idx`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ThrMma.thr_idx "Link to this definition")
```

```
`partition_A`(
```

```
`partition_B`(
```

```
`partition_C`(
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ThrMma._abc_impl "Link to this definition")
```

```
_`class`_`cutlass.cute.``ThrCopy`(
```

Bases: [`TiledCopy`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")

The thread Copy class for modeling a thread-slice of a tiled Copy.

```
`__init__`(
```

```
_`property`_`thr_idx`[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ThrCopy.thr_idx "Link to this definition")
```

```
`partition_S`(
```

```
`partition_D`(
```

```
`_abc_impl`_`=` `<_abc._abc_data` `object>`_[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ThrCopy._abc_impl "Link to this definition")
```

```
`cutlass.cute.``make_atom`(_`ty`_, _`values``=``None`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_atom "Link to this definition")
```

This is a wrapper around the _cute_ir.make_atom operation, providing default value for the values argument.

```
`cutlass.cute.``make_mma_atom`(
```

Makes an MMA Atom from an MMA Operation.

This function creates an MMA Atom from a given MMA Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters. They are not used as of today.

**Parameters:**
: **op** (_MmaOp_) – The MMA Operation to construct an Atom for

**Returns:**
: The MMA Atom

**Return type:**
: [MmaAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")

```
`cutlass.cute.``make_tiled_mma`(
```

Makes a tiled MMA from an MMA Operation or an MMA Atom.

**Parameters:**
: - **op_or_atom** (_Union__[__Op__,_[_MmaAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")_]_) – The MMA Operation or Atom
- **atom_layout_mnk** (_Layout_) – A Layout describing the tiling of Atom across threads
- **permutation_mnk** (_Tiler_) – A permutation Tiler describing the tiling of Atom across values including any permutation of such tiling

**Returns:**
: The resulting tiled MMA

**Return type:**
: [TiledMma](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

```
`cutlass.cute.``make_copy_atom`(
```

Makes a Copy Atom from a Copy Operation.

This function creates a Copy Atom from a given Copy Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters.

Example:

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
```

**Parameters:**
: - **op** (_CopyOp_) – The Copy Operation to construct an Atom for
- **copy_internal_type** (_Type__[__Numeric__]_) – An internal data type used to construct the source/destination layouts in unit of tensor elements

**Returns:**
: The Copy Atom

**Return type:**
: [CopyAtom](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

```
`cutlass.cute.``make_tiled_copy_tv`(
```

Create a tiled copy given separate thread and value layouts.

A TV partitioner is inferred based on the input layouts. The input thread layout
must be compact.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **thr_layout** (_Layout_) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
- **val_layout** (_Layout_) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy`(_`atom`_, _`layout_tv`_, _`tiler_mn`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy "Link to this definition")
```

Create a tiled type given a TV partitioner and tiler.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom, e.g. smit_copy and simt_async_copy, tma_load, etc.
- **layout_tv** (_Layout_) – Thread-value layout
- **tiler_mn** (_Tiler_) – Tile size
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_S`(_`atom`_, _`tiled_copy`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy_S "Link to this definition")
```

Create a tiled copy out of the copy_atom that matches the Src-Layout of tiled_copy.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **tiled_copy** ([_TiledCopy_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – Tiled copy
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_D`(_`atom`_, _`tiled_copy`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy_D "Link to this definition")
```

Create a tiled copy out of the copy_atom that matches the Dst-Layout of tiled_copy.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **tiled_copy** ([_TiledCopy_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) – Tiled copy
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_A`(_`atom`_, _`tiled_mma`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy_A "Link to this definition")
```

Create a tiled copy out of the copy_atom that matches the A-Layout of tiled_mma.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **tiled_mma** ([_TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – Tiled MMA
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_B`(_`atom`_, _`tiled_mma`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy_B "Link to this definition")
```

Create a tiled copy out of the copy_atom that matches the B-Layout of tiled_mma.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **tiled_mma** ([_TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – Tiled MMA
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_C`(_`atom`_, _`tiled_mma`_, _`*`_, _`loc``=``None`_, _`ip``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.make_tiled_copy_C "Link to this definition")
```

Create a tiled copy out of the copy_atom that matches the C-Layout of tiled_mma.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **tiled_mma** ([_TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – Tiled MMA
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for the partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

```
`cutlass.cute.``make_tiled_copy_C_atom`(
```

Create the smallest tiled copy that can retile LayoutC_TV for use with pipelined epilogues with subtiled stores.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom
- **mma** ([_TiledMma_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) – Tiled MMA
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

**Returns:**
: A tiled copy for partitioner

**Return type:**
: [TiledCopy](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")

**Raises:**
: **ValueError** – If the number value of CopyAtom’s source layout is greater than the size of TiledMma’s LayoutC_TV

```
`cutlass.cute.``make_cotiled_copy`(
```

Produce a TiledCopy from thread and value offset maps.
The TV Layout maps threads and values to the codomain of the data_layout.
It is verified that the intended codomain is valid within data_layout.
Useful when threads and values don’t care about owning specific coordinates, but
care more about the vector-width and offsets between them.

**Parameters:**
: - **atom** (_copy atom__,__e.g. simt_copy and simt_async_copy__,__tgen05.st__,__etc._)
- **atom_layout_tv** (_(__tid__,__vid__)__-> data addr_)
- **data_layout** (_data coord -> data addr_)
- **loc** (_source location for mlir__(__optional__)_)
- **ip** (_insertion point__(__optional__)_)

**Returns:**
: A tuple of A tiled copy and atom

**Return type:**
: tiled_copy

```
`cutlass.cute.``copy_atom_call`(
```

Execute a single copy atom operation.

The copy_atom_call operation executes a copy atom with the given operands.
Source and destination tensors have layout profile `(V)`.

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,

For src/dst layout like `(V, Rest...)`, the layout profile of `pred` must match `(Rest...)`.

> - Certain Atoms may require additional operation-specific keyword arguments.
> - Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
> for future releases.

Both `src` and `dst` operands are variadic, containing a variable number of tensors:

- For regular copy, `src` and `dst` each contain a single tensor.
- For copy with auxiliary operands, they contain the main tensor followed by
auxiliary tensors. For example:

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom specifying the transfer operation
- **src** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – Source tensor(s) with layout profile `(V)`. Can be a single Tensor
or a list/tuple of Tensors for operations with auxiliary source operands.
- **dst** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – Destination tensor(s) with layout profile `(V)`. Can be a single Tensor
or a list/tuple of Tensors for operations with auxiliary destination operands.
- **pred** (_Optional__[__Tensor__]__,__optional_) – Optional predication tensor for conditional transfers, defaults to None
- **loc** (_Any__,__optional_) – Source location information, defaults to None
- **ip** (_Any__,__optional_) – Insertion point, defaults to None
- **kwargs** (_Dict__[__str__,__Any__]_) – Additional copy atom specific arguments

**Raises:**
: **TypeError** – If source and destination element type bit widths differ

**Returns:**
: None

**Return type:**
: None

**Examples**:

```python
# Regular copy atom operation
cute.copy_atom_call(copy_atom, src, dst)

# Predicated copy atom operation
cute.copy_atom_call(copy_atom, src, dst, pred=pred)
```

```
`cutlass.cute.``mma_atom_call`(
```

Execute a single MMA atom operation.

The mma_atom_call operation executes an MMA atom with the given operands.
This performs a matrix multiplication and accumulation operation:
D = A * B + C

Note: The tensors ‘d’, ‘a’, ‘b’, and ‘c’ must only have a single fragment.

**Parameters:**
: - **atom** ([_MmaAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")) – The MMA atom to execute
- **d** (_Tensor_) – Destination tensor (output accumulator)
- **a** (_Tensor_) – First source tensor (matrix A)
- **b** (_Tensor_) – Second source tensor (matrix B)
- **c** (_Tensor_) – Third source tensor (input accumulator C)
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

Examples:

```python
# Call an MMA atom operation
cute.mma_atom_call(mma_atom, d_tensor, a_tensor, b_tensor, c_tensor)
```

```
`cutlass.cute.``gemm`(
```

The GEMM algorithm.

Computes `D <- A * B + C` where `C` and `D` can alias. Note that some MMA Atoms (e.g.
warpgroup-wide or tcgen05 MMAs) require manually setting an “accumulate” boolean field.

All tensors must be partitioned according to the provided MMA Atom.

For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
election internally. Manual thread selection is not required in such cases.

Following dispatch rules are supported:

- Dispatch [1]: (V) x (V) => (V)          => (V,1,1) x (V,1,1) => (V,1,1)
- Dispatch [2]: (M) x (N) => (M,N)        => (1,M,1) x (1,N,1) => (1,M,N)
- Dispatch [3]: (M,K) x (N,K) => (M,N)    => (1,M,K) x (1,N,K) => (1,M,N)
- Dispatch [4]: (V,M) x (V,N) => (V,M,N)  => (V,M,1) x (V,N,1) => (V,M,N)
- Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)

Operand flexibility:
- *a* and *b* can be a single Tensor (regular GEMM) or a sequence *[operand, scale_factor]* for block-scaled GEMM.

**Parameters:**
: - **atom** ([_MmaAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.MmaAtom "cutlass.cute.MmaAtom")) – MMA atom
- **d** (_Tensor_) – Destination tensor
- **a** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – First source tensor or sequence for advanced modes (e.g., *[a, sfa]*)
- **b** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – Second source tensor or sequence for advanced modes (e.g., *[b, sfb]*)
- **c** (_Tensor_) – Third source tensor
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point for MLIR, defaults to None
- **kwargs** (_dict_) – Additional keyword arguments

**Returns:**
: None

**Return type:**
: None

```
`cutlass.cute.``copy`(
```

Facilitates data transfer between two tensors conforming to layout profile `(V, Rest...)`.

**Parameters:**
: - **atom** ([_CopyAtom_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) – Copy atom specifying the transfer operation
- **src** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – Source tensor or list of tensors with layout profile `(V, Rest...)`
- **dst** (_Union__[__Tensor__,__List__[__Tensor__]__,__Tuple__[__Tensor__,__...__]__]_) – Destination tensor or list of tensors with layout profile `(V, Rest...)`
- **pred** (_Optional__[__Tensor__]__,__optional_) – Optional predication tensor for conditional transfers, defaults to None
- **loc** (_Any__,__optional_) – Source location information, defaults to None
- **ip** (_Any__,__optional_) – Insertion point, defaults to None
- **kwargs** (_Dict__[__str__,__Any__]_) – Additional copy atom specific arguments

**Raises:**
: - **TypeError** – If source and destination element type bit widths differ
- **ValueError** – If source and destination ranks differ
- **ValueError** – If source and destination mode-1 sizes differ
- **NotImplementedError** – If `V-mode` rank exceeds 2

**Returns:**
: None

**Return type:**
: None

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,
and src/dst layout like `((V, Rest...), Rest...)`

The algorithm recursively processes the `V-mode`, decomposing it until reaching the minimum granularity
compatible with the provided Copy Atom’s requirements.

Source and destination tensors must be partitioned in accordance with the Copy Atom specifications.
Post-partitioning, both tensors will exhibit a `(V, Rest...)` layout profile.

The operands *src* and *dst* are variadic, each containing a variable number of tensors:

- For regular copy, *src* and *dst* contain single source and destination tensors respectively.
- For copy with auxiliary operands, *src* and *dst* contain the primary tensors followed by
their respective auxiliary tensors.

**Precondition:** The size of mode 1 must be equal for both source and destination tensors:
`size(src, mode=[1]) == size(dst, mode=[1])`

**Examples**:

TMA copy operation with multicast functionality:

```python
cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask, cache_policy=policy)
```

Optional predication is supported through an additional tensor parameter. For partitioned tensors with
logical profile `((ATOM_V,ATOM_REST),REST,...)`, the predication tensor must maintain profile
compatibility with `(ATOM_REST,REST,...)`.

For Copy Atoms requiring single-threaded execution, thread election is managed automatically by the
copy operation. External thread selection mechanisms are not necessary.

> **Note**
>
> - Certain Atoms may require additional operation-specific keyword arguments.
> - Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
> for future releases.

```
`cutlass.cute.``basic_copy`(
```

Performs a basic element-wise copy.

This functions **assumes** the following pre-conditions:
1. *size(src) == size(dst)*

When the *src* and *dst* shapes are static, the pre-conditions are actually verified and the
element-wise loop is fully unrolled.

**Parameters:**
: - **src** (_Tensor_) – Source tensor
- **dst** (_Tensor_) – Destination tensor
- **loc** (_Optional__[__Location__]__,__optional_) – Source location for MLIR, defaults to None
- **ip** (_Optional__[__InsertionPoint__]__,__optional_) – Insertion point, defaults to None

```
`cutlass.cute.``basic_copy_if`(
```

Performs a basic predicated element-wise copy.

This functions **assumes** the following pre-conditions:
1. *size(src) == size(dst)*
2. *size(src) == size(pred)*

When all shapes are static, the pre-conditions are actually verified and the element-wise loop
is fully unrolled.

```
`cutlass.cute.``autovec_copy`(
```

Auto-vectorization SIMT copy policy.

Given a source and destination tensors that are statically shaped, this policy figures out the
largest safe vector width that the copy instruction can take and performs the copy.

```
`cutlass.cute.``prefetch`(
```

The Prefetch algorithm.

The “prefetch” expects source tensors to be partitioned according to the provided Copy Atom.
Prefetch is used for loading tensors from global memory to L2.

Prefetch accepts Copy Atom but not all are allowed. Currently, only supports TMA prefetch.

```python
cute.prefetch(tma_prefetch, src)
```

For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
election internally. Manual thread selection is not required in such cases.

```
`cutlass.cute.``acos`(
```

Compute element-wise arc cosine of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the arc cosine of each element in input tensor

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = acos(y)  # Compute arc cosine
```

```
`cutlass.cute.``asin`(
```

Compute element-wise arc sine of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the arc sine of each element in input tensor

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = asin(y)  # Compute arc sine
```

```
`cutlass.cute.``atan`(
```

Compute element-wise arc tangent of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the arc tangent of each element in input tensor

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = atan(y)  # Compute arc tangent
```

```
`cutlass.cute.``atan2`(
```

Compute element-wise arc tangent of two tensors.

Computes atan2(a, b) element-wise. The function atan2(a, b) is the angle in radians
between the positive x-axis and the point given by the coordinates (b, a).

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – First input tensor (y-coordinates)
- **b** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Second input tensor (x-coordinates)
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the arc tangent of a/b element-wise

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
y = cute.make_rmem_tensor(ptr1, layout).load()  # y coordinates
x = cute.make_rmem_tensor(ptr2, layout).load()  # x coordinates
theta = atan2(y, x)  # Compute angles
```

```
`cutlass.cute.``cos`(
```

Compute element-wise cosine of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor (in radians)
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the cosine of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = cos(y)  # Compute cosine
```

```
`cutlass.cute.``erf`(
```

Compute element-wise error function of the input tensor.

The error function is defined as:
erf(x) = 2/√π ∫[0 to x] exp(-t²) dt

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the error function value for each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = erf(y)  # Compute error function
```

```
`cutlass.cute.``exp`(
```

Compute element-wise exponential of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the exponential of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = exp(y)  # Compute exponential
```

```
`cutlass.cute.``exp2`(
```

Compute element-wise base-2 exponential of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing 2 raised to the power of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = exp2(y)  # Compute 2^x
```

```
`cutlass.cute.``log`(
```

Compute element-wise natural logarithm of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the natural logarithm of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log(y)  # Compute natural logarithm
```

```
`cutlass.cute.``log10`(
```

Compute element-wise base-10 logarithm of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the base-10 logarithm of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log10(y)  # Compute log base 10
```

```
`cutlass.cute.``log2`(
```

Compute element-wise base-2 logarithm of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the base-2 logarithm of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log2(y)  # Compute log base 2
```

```
`cutlass.cute.``rsqrt`(
```

Compute element-wise reciprocal square root of the input tensor.

Computes 1/√x element-wise.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the reciprocal square root of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = rsqrt(y)  # Compute 1/√x
```

```
`cutlass.cute.``sin`(
```

Compute element-wise sine of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor (in radians)
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the sine of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = sin(y)  # Compute sine
```

```
`cutlass.cute.``sqrt`(
```

Compute element-wise square root of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the square root of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = sqrt(y)  # Compute square root
```

```
`cutlass.cute.``tan`(
```

Compute element-wise tangent of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor (in radians)
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the tangent of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = tan(y)  # Compute tangent
```

```
`cutlass.cute.``tanh`(
```

Compute element-wise hyperbolic tangent of the input tensor.

**Parameters:**
: - **a** (_Union__[_[_TensorSSA_](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA")_,__Numeric__]_) – Input tensor
- **fastmath** (_bool__,__optional_) – Enable fast math optimizations, defaults to False

**Returns:**
: Tensor containing the hyperbolic tangent of each element

**Return type:**
: Union[[TensorSSA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.TensorSSA "cutlass.cute.TensorSSA"), Numeric]

Example:

```console
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = tanh(y)  # Compute hyperbolic tangent
```

```
_`class`_`cutlass.cute.``ffi`(_`*`_, _`name``:` `str`_, _`params_types``:` `list` `=` `[]`_, _`return_type``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi "Link to this definition")
```

Bases: `object`

Foreign Function Interface (FFI) wrapper for external function invocation in the CUTLASS Python DSL.

This class enables calling external MLIR function prototypes from Python code, handling type conversion,
prototype registration, and dynamic insertion of function symbols into MLIR modules as needed.

**Parameters:**
: - **name** (_str_) – Name of the external function. This will be used as the symbol name when calling or registering a prototype in the MLIR module.
- **params_types** (_list__,__optional_) – List of argument types for the external function. These can be CUTLASS numeric types, numeric meta types, or types convertible via *get_mlir_types*.
- **return_type** (_optional_) – The return type of the external function. If not specified, the function is assumed to have no return value.

```
`__call__`(_`*``args`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi.__call__ "Link to this definition")
```

Calls the external function with the given arguments, ensuring argument and result types match the prototype.

```
`__init__`(_`*`_, _`name``:` `str`_, _`params_types``:` `list` `=` `[]`_, _`return_type``=``None`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi.__init__ "Link to this definition")
```

```
`_get_prototype_region`(_`current_op`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi._get_prototype_region "Link to this definition")
```

Helper method to determine the appropriate MLIR module and region for inserting a function prototype.

This method recursively traverses the current operation’s parent hierarchy to find the correct module
and region where the function prototype should be inserted. It supports both builtin.module and gpu.module.
:param current_op: The current operation to check.
:type current_op: Operation

**Returns:**
: A tuple containing the module operation and the insertion region.

**Return type:**
: tuple

```
_`static`_`_to_mlir_types`(_`args`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi._to_mlir_types "Link to this definition")
```

Helper method to convert a list of arguments to their corresponding MLIR types.

This method converts CUTLASS numeric types, numeric meta types, and types convertible via *get_mlir_types*
to their corresponding MLIR types.
:param args: The list of arguments to convert to MLIR types.
:type args: list

**Returns:**
: A list of MLIR types.

**Return type:**
: list

```
_`static`_`_type_check`(_`callee`_, _`exec_types`_, _`returns_types`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi._type_check "Link to this definition")
```

Helper method to check if the function prototype types match the expected types.

This method compares the input and output types of the function prototype with the provided expected types.
:param callee: The function prototype operation to check.
:type callee: func.FuncOp
:param exec_types: The expected input types.
:type exec_types: list
:param returns_types: The expected output types.
:type returns_types: list

```
`_create_prototype_in_region`(_`op`_, _`region`_, _`exec_args`_)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/#cutlass.cute.ffi._create_prototype_in_region "Link to this definition")
```

Helper method to create or retrieve a function prototype in the current module.

This method checks if a function prototype with the given name already exists in the symbol table of the current module.
If it does, it checks if the prototype’s types match the expected types. If it does not, it raises an error.
If it does not exist, it creates a new function prototype and inserts it into the current region.
:param op: The module operation to check.
:type op: Operation
:param region: The region to insert the function prototype into.
:type region: Region
:param exec_args: The arguments to pass to the function prototype.
:type exec_args: list
