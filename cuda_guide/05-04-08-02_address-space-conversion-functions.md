---
title: "5.4.8.2. Address Space Conversion Functions"
section: "5.4.8.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#address-space-conversion-functions"
---

### [5.4.8.2. Address Space Conversion Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#address-space-conversion-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#address-space-conversion-functions "Permalink to this headline")

CUDA pointers (`T*`) can access objects regardless of where the objects are stored. For example, an `int*` can access `int` objects whether they reside in global or shared memory.

Address space conversion functions are used to convert between generic addresses and addresses in specific address spaces.
These functions are useful when the compiler cannot determine a pointer’s address space, for example, when crossing translation units or interacting with PTX instructions.

```cuda
__device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
__device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
__device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
__device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local
```

```cuda
__device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
__device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
__device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
__device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local
```

As an example of inter-operating with PTX instructions, the `ld.shared.s32 r0, [ptr];` PTX instruction expects `ptr` to refer to the shared memory address space.
A CUDA program with an `int*` pointer to an object in `__shared__` memory needs to convert this pointer to the shared address space before passing it to the PTX instruction by calling `__cvta_generic_to_shared` as follows:

```cuda
__shared__ int smem_var;
smem_var        = 42;
size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
int    output;
asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
assert(output == 42);
```

A common optimization that exploits these address representations is reducing data structure size by leveraging the fact that the address ranges of shared, local, and constant spaces are smaller than 32 bits, which allows storing 32-bit addresses instead of 64-bit pointers and save registers. Additionally, 32-bit arithmetic is faster than 64-bit arithmetic.
To obtain the 32-bit integer representation of these addresses, truncate the 64-bit value to 32 bits by casting from an unsigned 64-bit integer to an unsigned 32-bit integer:

```cuda
__shared__ int smem_var;
uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));
```

To recover a generic address from such a 32-bit representation, zero-extend the address back to an unsigned 64-bit integer and then call the corresponding address space conversion function:

```cuda
size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
assert(generic_ptr == &smem_var);
```

---
