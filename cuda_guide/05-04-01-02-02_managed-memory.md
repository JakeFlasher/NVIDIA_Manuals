---
title: "5.4.1.2.2. __managed__ Memory"
section: "5.4.1.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#managed-memory"
---

#### [5.4.1.2.2. __managed__ Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#managed-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#managed-memory "Permalink to this headline")

`__managed__` variables have the following restrictions:

- The address of a `__managed__` variable is not a constant expression.
- A `__managed__` variable shall not have a reference type `T&`.
- The address or value of a `__managed__` variable shall not be used when the CUDA runtime may not be in a valid state, including the following cases:
  - In static/dynamic initialization or destruction of an object with `static` or `thread_local` storage duration.
  - In code that executes after `exit()` has been called. For example, a function marked with `__attribute__((destructor))`.
  - In code that executes when the CUDA runtime may not be initialized. For example, a function marked with `__attribute__((constructor))`.
- A `__managed__` variable cannot be used as an unparenthesized id-expression argument to a `decltype()` expression.
- `__managed__` variables have the same coherence and consistency behavior as specified for [dynamically allocated managed memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory).
- See also the restrictions for [local variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#local-variables).

Here are examples of legal and illegal uses of `__managed__` variables:

```cuda
#include <cassert>

__device__ __managed__ int global_var = 10; // OK

int* ptr = &global_var;                     // ERROR: use of a managed variable in static initialization

struct MyStruct1 {
    int field;
    MyStruct1() : field(global_var) {};
};

struct MyStruct2 {
    ~MyStruct2() { global_var = 10; }
};

MyStruct1 temp1; // ERROR: use of managed variable in dynamic initialization

MyStruct2 temp2; // ERROR: use of managed variable in the destructor of
                 //        object with static storage duration

__device__ __managed__ const int const_var = 10;         // ERROR: const-qualified type

__device__ __managed__ int&      reference = global_var; // ERROR: reference type

template <int* Addr>
struct MyStruct3 {};

MyStruct3<&global_var> temp;     // ERROR: address of managed variable is not a constant expression

__global__ void kernel(int* ptr) {
    assert(ptr == &global_var);  // OK
    global_var = 20;             // OK
}

int main() {
    int* ptr = &global_var;      // OK
    kernel<<<1, 1>>>(ptr);
    cudaDeviceSynchronize();
    global_var++;                // OK
    decltype(global_var) var1;   // ERROR: managed variable used as unparenthesized argument to decltype

    decltype((global_var)) var2; // OK
}
```
