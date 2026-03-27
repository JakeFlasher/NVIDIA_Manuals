---
title: "Array"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#array"
---

### [Array](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#array)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#array "Permalink to this headline")

```c++
template <
  typename T,       // element type
  int N             // number of elements
>
struct Array;
```

`Array<class T, int N>` defines a statically sized array of elements of type _T_ and size _N_. This class is similar to
[`std::array<>`](https://en.cppreference.com/w/cpp/container/array) in the Standard Library with one notable exception:
partial specializations exist to pack or unpack elements smaller than one byte.

`Array<>` is intended to be a convenient and uniform container class to store arrays of numeric elements regardless of data type or vector length. The storage needed is expected to be the minimum necessary given the logical size of each numeric type in bits (numeric types smaller than one byte are densely packed). Nevertheless, the size reported by `sizeof(Array<T, N>)` is always an integer multiple of bytes.

Storing numeric elements in a C++ STL-style container class enables useful modern C++ mechanisms such as range-based for loops. For example, to print the elements of `Array<>`, the following range-based for loop syntax is always valid regardless of numeric data type, compute capability, or context in host or device code.

Example:

```c++
int const kN;
Array<T, kN> elements;

CUTLASS_PRAGMA_UNROLL                        // required to ensure array remains in registers
for (auto x : elements) {
  printf("%d, %f", int64_t(x), double(x));   // explictly convert to int64_t or double
}
```

When copying `Array<>` objects or passing them as arguments to methods, it is best to avoid accessing individual elements. This enables the use of vector instructions to perform the operation more efficiently. For example, setting all elements to zero is best performed by calling the `clear()` method. Copies should be performed by assigning the entire object.

Example:

```c++
#include <cutlass/array.h>

int const kN;
Array<T, kN> source;
Array<T, kN> destination;

source.clear();         // set all elements to value of zero

destination = source;   // copy to `destination`
```

`Array<>` may be used to store elements smaller than one byte such as 4b integers.

```c++
Array<int4b_t, 2> packed_integers;

static_assert(
  sizeof(packed_integers) == 1,
 "Packed storage of sub-byte data types is compact.");

// Access array elements using usual indirection and assignment operators
packed_integers[0] = 2_s4;
packed_integers[1] = 3_s4;

CUTLASS_PRAGMA_UNROLL
for (auto x : elements) {
  printf("%d", int(x));       // access elements normally
}
```
