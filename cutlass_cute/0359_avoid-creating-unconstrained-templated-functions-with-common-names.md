---
title: "Avoid creating unconstrained templated functions with common names"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#avoid-creating-unconstrained-templated-functions-with-common-names"
---

#### [Avoid creating unconstrained templated functions with common names](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#avoid-creating-unconstrained-templated-functions-with-common-names)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#avoid-creating-unconstrained-templated-functions-with-common-names "Permalink to this headline")

See [C++ Core Guidelines T.47](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#t47-avoid-highly-visible-unconstrained-templates-with-common-names):
“Avoid highly visible unconstrained templates
with common names.”
Argument-dependent lookup (ADL) means that
if users call a function name without specifying the namespace,
the compiler can find overloads
of that function in any namespace.
This can lead to ambiguous overloads in users’ code,
just because they happened to include one of your header files
that exposes an unconstrained function template.
The following illustrates this
with an unconstrained swap overload in the `cutlass` namespace.

```c++
#include <cassert>
#include <memory>
#include <utility>

// Uncomment the line below to observe unwarranted build errors.
//#define BAD_CUTLASS_SWAP 1

namespace cutlass {
struct Bar {
  float f;
};
} // namespace cutlass

#ifdef BAD_CUTLASS_SWAP
namespace cutlass {

// don't do this
template<class T>
void swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

} // namespace cutlass
#endif // BAD_CUTLASS_SWAP

namespace other {

#ifdef BAD_CUTLASS_SWAP
using cutlass::swap;
#endif // BAD_CUTLASS_SWAP

// Imagine for the sake of this example
// that "foo" is a less common name,
// and that T is constrained via
// std::enable_if or a requires clause.
template<class T>
void foo(T& a, T& b) {
  // The usual idiom for using std::swap is the "swap two-step":
  //
  // 1. import std::swap into the current scope, then
  // 2. call swap without namespace qualification.
  //
  // That won't build if we have another swap
  // overload available in the scope already.

  using std::swap;
  swap(a, b); // OBSERVE UNWARRANTED BUILD ERROR HERE
}

} // namespace other

int main() {
  int x = 42;
  int y = 43;
  other::foo(x, y);
  assert(x == 43);
  assert(y == 42);

  cutlass::Bar a{42.0};
  cutlass::Bar b{43.0};
  other::foo(a, b);
  assert(a.f == 43.0);
  assert(b.f == 42.0);

  // GCC 7.5 std::unique_ptr::reset calls swap,
  // leading to the same issue as above.
  // GCC 12.2's implementation of std::unique_ptr
  // does not have this issue.  Nevertheless,
  // breaking the swap two-step will break users' code,
  // just by them happening to include your headers.
  auto ptr = std::make_unique<cutlass::Bar>(cutlass::Bar{666.0f});
  ptr.reset(new cutlass::Bar{777.0f}); // OBSERVE UNWARRANTED BUILD ERROR HERE

  return 0;
}
```
