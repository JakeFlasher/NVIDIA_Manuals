---
title: "5.3.9.6.4. Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions"
section: "5.3.9.6.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#implicitly-declared-and-non-virtual-explicitly-defaulted-functions"
---

#### [5.3.9.6.4. Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#implicitly-declared-and-non-virtual-explicitly-defaulted-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#implicitly-declared-and-non-virtual-explicitly-defaulted-functions "Permalink to this headline")

Implicitly-declared special member functions are those the compiler declares for a class when the user does not declare them; Explicitly-defaulted functions are ones the user declares but marks with `= default`. The special member functions that are implicitly-declared or explicitly-defaulted are default constructor, copy constructor, move constructor, copy assignment operator, move assignment operator, and destructor.

Let `F` denote a non-`virtual` function that is either implicitly declared or explicitly defaulted on its first declaration.
The execution space specifiers for `F` are the union of the execution space specifiers of all functions that invoke it. Note that for this analysis, a `__global__` caller will be treated as a `__device__` caller. For example:

```cuda
class Base {
    int x;
public:
    __host__ __device__ Base() : x(10) {}
};

class Derived : public Base {
    int y;
};

class Other: public Base {
    int z;
};

__device__ void foo() {
    Derived D1;
    Other D2;
}

__host__ void bar() {
    Other D3;
}
```

In this case, the implicitly declared constructor function `Derived::Derived()` will be treated as a `__device__` function because it is only invoked from the `__device__` function `foo()`. The implicitly declared constructor function `Other::Other()` will be treated as a `__host__ __device__` function since it is invoked both from both a `__device__` function `foo()` and a `__host__` function `bar()`.

Additionally, if `F` is an implicitly-declared `virtual` function (for example, a `virtual` destructor), the execution spaces of each virtual function `D` that is overridden by `F` are added to the set of execution spaces for `F` if `D` not implicitly-declared.

For example:

```cuda
struct Base1 {
    virtual __host__ __device__ ~Base1() {}
};

struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                            // ~Derived1() has __host__ __device__  execution space specifiers

struct Base2 {
    virtual __device__ ~Base2() = default;
};

struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                            // ~Derived2() has __device__ execution space specifiers
```
