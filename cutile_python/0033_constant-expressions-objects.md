---
title: "Constant Expressions & Objects"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#constant-expressions-objects"
---

### [Constant Expressions & Objects](https://docs.nvidia.com/cuda/cutile-python#constant-expressions-objects)[](https://docs.nvidia.com/cuda/cutile-python/#constant-expressions-objects "Permalink to this headline")

Some facilities require certain parameters to be an object that is known statically at compilation time.
_Constant expressions_ produce _constant objects_ suitable for such parameters. Constant expressions are:

- A literal object.
- Integer arithmetic expressions where all the operands are literal objects.
- A local object or parameter that is assigned from a literal object or constant expression.
- A global object that is defined at the time of compilation or launch.

By default, numeric constants are _loosely typed_: until used in a context that requires
a type of a specific width, integer constants have infinite precision, and floating-point
constants are stored in the IEEE 754 double precision format.

A _strictly typed_ constant can be created by calling a dtype object as a constructor,
e.g. `ct.int16(5)` creates a strictly typed `int16` constant. When a strictly typed constant
is combined with a loosely typed constant, the result is a strictly typed constant.
For example `ct.int16(5) + 2` will create a strictly typed `int16` constant 7.

Combining two strictly typed constants creates a new strictly typed constant. In this case,
the regular [type promotion](https://docs.nvidia.com/cuda/cutile-python/data.html#data-arithmetic-promotion) rules apply.
For example, `ct.int16(5) + ct.int32(7)` will create a strictly typed `int32` constant 12.
