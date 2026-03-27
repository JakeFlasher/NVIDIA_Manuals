---
title: "5.3.9.6.5. Polymorphic Classes"
section: "5.3.9.6.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#polymorphic-classes"
---

#### [5.3.9.6.5. Polymorphic Classes](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#polymorphic-classes)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#polymorphic-classes "Permalink to this headline")

Polymorphic classes, namely those with `virtual` functions, derived from other polymorphic classes, or with polymorphic data members, are subject to the following restrictions:

- Copying polymorphic objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.
- The execution space of an overridden `virtual` function must match the execution space of the function in the base class.

Example:

```cuda
struct MyClass {
    virtual __host__ __device__ void f() {}
};

__global__ void kernel(MyClass my_class) {
    my_class.f(); // undefined behavior
}

int main() {
    MyClass my_class;
    kernel<<<1, 1>>>(my_class);
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/To39sGTrW).

---

```cuda
struct BaseClass {
    virtual __host__ __device__ void f() {}
};

struct DerivedClass : BaseClass {
    __device__ void f() override {} // ERROR
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/xfKhEGfdG).
