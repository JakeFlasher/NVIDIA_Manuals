---
title: "Programming Model"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html#programming-model"
---

## [Programming Model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#programming-model)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#programming-model "Permalink to this headline")

****CuTe Layout Algebra Only support 32bit****

  Today, we only support 32bit shapes/strides in CuTe layouts. 64bit or arbitrary
  width support is planned for future releases.

****Python Native Data Types****

  CuTe DSL supports Python data structures when used for “meta-programming,”
  but these structures cannot be treated as dynamic values modifiable at runtime.
  For instance, lists and dictionaries can be used to configure kernel parameters
  during compilation or serve as containers for dynamic values,
  but their structure and organization cannot be altered during kernel execution.

  - ****Static Values:****
    - Evaluated during JIT compilation phase
    - Immutable after compilation completes
    - Most Python native types (lists, tuples, dictionaries) are processed as static values
    - Primarily utilized for “meta-programming” and configuration purposes
    - Example: Lists can contain dynamic values but their structure cannot
    be modified during kernel execution
  - ****Dynamic Values:****
    - Evaluated during runtime execution
    - Modifiable during execution of JIT-compiled functions
    - Only a specific subset of Python types are supported as dynamic values
    - Primitive types are automatically converted when passed as function arguments:
      - `int` → `Int32` (may be updated to `Int64` in future releases)
      - `bool` → `Bool`
      - `float` → `Float32` (may be updated to `Float64` in future releases)

  The JIT compiler processes Python native types analogously to C++ template parameters.
  The compiled code cannot manipulate dynamic values of composite types
  such as lists, tuples, or dictionaries.

  For example, following code doesn’t work as traditional Python program inside JIT function.

  ```python
  @cute.jit
  def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
      xs = [a, b]
      # indexing list with dynamic index is not supported in CuTe DSL:
      res[0] = xs[i]

      if i == 0:
          # This will alway append Float32(3.0) to the list regardless
          # of the runtime value of `i`
          xs.append(Float32(3.0))

      for i in range(10):
          # This only append one element to the list at compile-time
          # as loop doesn't unroll at compile-time
          xs.append(Float32(1.0))
  ```

****Python Function****

  The DSL currently has **limited support for return values** from Python functions.
  At the moment, only `constexpr` values can be returned, while returning **dynamic values** is **not yet supported**.
  This capability is planned for a future release.

  Example:

  ```python
  @cute.jit
  def baz(a: cutlass.Constexpr):
      return a + 1

  @cute.jit
  def foo(a: cutlass.Int32):
      return a + 1

  @cute.jit
  def bar(a: cutlass.Int32):
      val = foo(a)  # works

  val = baz(10)   # works
  val = bar(10)   # works
  foo(10)         # currently unsupported in CuTe DSL
  ```

****Expression or Statement with Dependent Type****

  CuTe DSL implements static typing and does not support dependent types.
  The type of each expression must be determinable during compile time,
  in contrast to standard Python which implements dynamic typing.

  Example illustrating functionality in Python that is not supported in the DSL:

  ```python
  # Valid in standard Python, but unsupported in CuTe DSL
  max(int(1), float(2.0))  # => 2.0 : float
  max(int(3), float(2.0))  # => 3   : int
  ```

  In CuTe DSL, types are promoted. For example:

  ```python
  @cute.jit
  def foo(a: Int32, b: Float32, res: cute.Tensor):
      res[0] = max(a, b)  # Type is automatically promoted to Float32
  ```

  Following code using inlined if-else expression with dependent types
  is not supported in CuTe DSL:

  ```python
  @cute.jit
  def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
      res[0] = a if cond else b
  ```

****Control Flow****

  The DSL transforms Python control flow statements (`if`, `for`, `while`)
  during Abstract Syntax Tree (AST) processing into structured control flow in MLIR
  which has the same constraints as dependent types. For instance,
  changing type of a variable in loop body is not allowed.

  - Variables must be defined prior to the control flow statement
  - Type consistency must be maintained throughout the control flow statement
  - Don’t support early exit or return from if-else statements

  Example illustrating functionality in Python that is not supported in the DSL:

  ```python
  @cute.jit
  def foo():
      a = Int32(1)
      for i in range(10):
          a = Float32(2)  # Changing type inside loop-body is not allowed in the DSL
  ```

****Built-in Operators****

  The DSL transforms built-in operators like `and`, `or`, `max`, `min`, etc.
  into MLIR operations. They also follow the same constraints of dependent types.
  For instance, `a and b` requires `a` and `b` to be of the same type.

****Special Variables****

  The DSL treats `_` as a special variable that it’s value is meant to be ignored.
  It is not allowed to read `_` in the DSL.

  Example illustrating functionality in Python that is not supported in the DSL:

  ```python
  @cute.jit
  def foo():
      _ = 1
      print(_)  # This is not allowed in the DSL
  ```

****Object Oriented Programming****

  The DSL is implemented on top of Python and supports Python’s object-oriented programming (OOP) features
  for meta-programming at compile-time.

  However, similar to other composed data types, the DSL provides limited support for OOP when objects
  contain dynamic values. It is strongly recommended to avoid passing dynamic values between member methods
  through class state in your code.

  The following example illustrates functionality in Python that is not supported in the DSL
  without implementing the `DynamicExpression` protocol:

  ```python
  class Foo:
      def __init__(self, a: Int32):
          self.a = a

      def set_a(self, i: Int32):
          self.a = i

      def get_a(self):
          return self.a

  @cute.jit
  def foo(a: Int32, res: cute.Tensor):
      foo = Foo(a)
      for i in range(10):
          foo.set_a(i)

      # This fails to compile because `a` is assigned a local value defined within the for-loop body
      # and is not visible outside of the loop body
      res[0] = foo.get_a()
  ```

  The example above fails to compile because `Foo.a` is assigned a local value defined within the for-loop body,
  which is not visible outside the loop body.

  The CuTe DSL implements an internal mechanism that provides limited support for OOP patterns via protocol.
  As the DSL continues to evolve to support additional features, this mechanism is subject to change
  and is not recommended for direct use in users’ code for better portability.

****CuTe Layout algebra in native Python****

  Entirety of CuTe Layout algebra operations and APIs require JIT compilation. These
  functionalities are exclusively available within JIT-compiled functions and cannot be
  accessed in standard Python execution environments.

  Additionally, there exists a restricted set of data types that can be passed as arguments
  to JIT-compiled functions, which further constrains their usage in native Python contexts.
  Only following CuTe algebra types are supported as JIT function arguments: `Tensor`, `Pointer`,
  `Shape`, `Stride`, `Coord` and `IntTuple`. For `Stride`, we don’t support `ScacledBasis`
  from native Python Context. Unfortunately, in the first release, we don’t support
  passing `Layout` under native Python Context.
