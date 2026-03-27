---
title: "5.3.10.3. constexpr Functions"
section: "5.3.10.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#constexpr-functions"
---

### [5.3.10.3. constexpr Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#constexpr-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#constexpr-functions "Permalink to this headline")

By default, a `constexpr` function cannot be called from a function with incompatible execution space, in the same way as standard functions.

- Calling a device-only `constexpr` function from a host-function during host code generation phase, namely when `__CUDA_ARCH__` macro is undefined. Example:

```cuda
constexpr __device__ int device_function() { return 0; }
int main() {
    int x = device_function();  // ERROR, calling a device-only constexpr function from host code
}
```
- Calling a host-only `constexpr` function from a `__device__` or `__global__` function, during device code generation phase, namely when `__CUDA_ARCH__` macro is defined. Example:

```cuda
constexpr int host_function() { return 0; }
__device__ void device_function() {
    int x = host_function();  // ERROR, calling a host-only constexpr function from device code
}
```

Note that a function template specialization may not be a `constexpr` function even if the corresponding template function is marked with the keyword `constexpr`.

**Relaxed constexpr-Function Support**

The experimental `nvcc` flag `--expt-relaxed-constexpr`  can be used to relax this constraint for both `__host__` and `__device__` functions. However, a `__global__` function cannot be declared as `constexpr`. `nvcc` will also define the macro `__CUDACC_RELAXED_CONSTEXPR__`.

When this flag is specified, the compiler will support cross execution space calls described above, as follows:

1. A call to a `constexpr` function in a cross-execution space is supported if it occurs in a context that requires constant evaluation, such as the initializer of a `constexpr` variable. Example:

```cuda
constexpr __host__ int host_function(int x) { return x + 1; };
__global__ void kernel() {
    constexpr int val = host_function(1); // CORRECT, the call is in a context that requires constant evaluation.
}
constexpr __device__ int device_function(int x) { return x + 1; }
int main() {
    constexpr int val = device_function(1); // CORRECT, the call is in a context that requires constant evaluation.
}
```
2. Device code is generated during device code generation for the body of a host-only constexpr function, unless it is not used or is only called in a `constexpr` context. Example:

```cuda
// NOTE: "host_function" is emitted in generated device code because
//       it is called from device code in a non-constexpr context
constexpr int host_function(int x) { return x + 1; }
__device__ int device_function(int in) {
    return host_function(in);  // CORRECT, even though argument is not a constant expression
}
```
3. All code restrictions that apply to a device function also apply to the `constexpr` host-only function called from the device code. However, the compiler may not emit any build-time diagnostics for restrictions related to the compilation process.

For example, the following code patterns are not supported in the body of the host function. This is similar to any device function; however, no compiler diagnostic may be generated.
  - One-Definition Rule (ODR)-use of a host variable or host-only non-`constexpr` function. Example:

```cuda
int host_var1, host_var2;
constexpr int* host_function(bool b) { return b ? &host_var1 : &host_var2; };
__device__ int device_function(bool flag) {
    return *host_function(flag); // ERROR, host_function() attempts to refer to the host variables
                                 //        'host_var1' and 'host_var2'.
                                 //        The code will compile, but will NOT execute correctly.
}
```
  - Use of exceptions `throw/catch` and Run-Time Type Information `typeid/dynamic_cast`. Example:

```cuda
struct Base { };
struct Derived : public Base { };
// NOTE: "host_function" is emitted in generated device code
constexpr int host_function(bool b, Base *ptr) {
    if (b) {
        return 1;
    }
    else if (typeid(ptr) == typeid(Derived)) { // ERROR, use of typeid in code executing on the GPU
        return 2;
    }
    else {
        throw int{4}; // ERROR, use of throw in code executing on the GPU
    }
}
__device__ void device_function(bool flag) {
    Derived d;
    int val = host_function(flag, &d); //ERROR, host_function() attempts use typeid and throw(),
                                       //       which are not allowed in code that executes on the GPU
}
```
4. During host code generation, the body of a device-only `constexpr` function is preserved in the code sent to the host compiler. However, if the body of a device function attempts to ODR-use a namespace-scope device variable or a non-`constexpr` device function, the call to the device function from host code is not supported. While the code may build without compiler diagnostics, it may behave incorrectly at runtime. Example:

```cuda
__device__ int device_var1, device_var2;
constexpr __device__ int* device_function(bool b) { return b ? &device_var1 : &device_var2; };
int host_function(bool flag) {
    return *device_function(flag); // ERROR, device_function() attempts to refer to device variables
                                   //        'device_var1' and 'device_var2'
                                   // The code will compile, but will NOT execute correctly.
}
```

> **Warning**
>
> Due to the above restrictions and the lack of compiler diagnostics for incorrect usage, it is recommended to avoid calling a function in the Standard C++ headers `std::` from device code. The implementation of such functions varies depending on the host platform. Instead, it is strongly suggested to call the equivalent functionality in the CUDA C++ Standard Library [libcu++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#libcu), in the `cuda::std::` namespace.
