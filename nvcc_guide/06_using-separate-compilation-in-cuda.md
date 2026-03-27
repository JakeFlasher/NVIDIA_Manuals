---
title: "6. Using Separate Compilation in CUDA"
section: "6"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-separate-compilation-in-cuda"
---

# [6. Using Separate Compilation in CUDA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#using-separate-compilation-in-cuda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-separate-compilation-in-cuda "Permalink to this headline")

Prior to the 5.0 release, CUDA did not support separate compilation, so CUDA code could not call device functions or access variables across files. Such compilation is referred to as _whole program compilation_. We have always supported the separate compilation of host code, it was just the CUDA device code that needed to all be within one file. Starting with CUDA 5.0, separate compilation of device code is supported, but the old whole program mode is still the default, so there are new options to invoke separate compilation.

## [6.1. Code Changes for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#code-changes-for-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#code-changes-for-separate-compilation "Permalink to this headline")

The code changes required for separate compilation of device code are the same as what you already do for host code, namely using `extern` and `static` to control the visibility of symbols. Note that previously `extern` was ignored in CUDA code; now it will be honored. With the use of `static` it is possible to have multiple device symbols with the same name in different files. For this reason, the CUDA API calls that referred to symbols by their string name are deprecated; instead the symbol should be referenced by its address.

## [6.2. NVCC Options for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-options-for-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-options-for-separate-compilation "Permalink to this headline")

CUDA works by embedding device code into host objects. In whole program compilation, it embeds executable device code into the host object. In separate compilation, we embed relocatable device code into the host object, and run `nvlink`, the device linker, to link all the device code together. The output of nvlink is then linked together with all the host objects by the host linker to form the final executable.

The generation of relocatable vs executable device code is controlled by the `--relocatable-device-code` option.

The `--compile` option is already used to control stopping a compile at a host object, so a new option `--device-c` is added that simply does `--relocatable-device-code=true --compile`.

To invoke just the device linker, the `--device-link` option can be used, which emits a host object containing the embedded executable device code. The output of that must then be passed to the host linker. Or:

```text
nvcc <objects>
```

can be used to implicitly call both the device and host linkers. This works because if the device linker does not see any relocatable code it does not do anything.

The following figure shows the flow.

![CUDA Separate Compilation Trajectory](images/using-separate-compilation-in-cuda_1.png)

CUDA Separate Compilation Trajectory

There is also a `-r` option for doing a relocatable link that generates a new relocatable object containing relocatable device code.

## [6.3. Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#libraries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libraries "Permalink to this headline")

The device linker has the ability to read the static host library formats (`.a` on Linux and Mac OS X, `.lib` on Windows). It ignores any dynamic (`.so` or `.dll`) libraries. The `--library` and `--library-path` options can be used to pass libraries to both the device and host linker. The library name is specified without the library file extension when the `--library` option is used.

```text
nvcc --gpu-architecture=sm_100 a.o b.o --library-path=<path> --library=foo
```

Alternatively, the library name, including the library file extension, can be used without the `--library` option on Windows.

```text
nvcc --gpu-architecture=sm_100 a.obj b.obj foo.lib --library-path=<path>
```

Note that the device linker ignores any objects that do not have relocatable device code.

## [6.4. Examples](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#examples)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#examples "Permalink to this headline")

Suppose we have the following files:

```c++
//---------- b.h ----------
#define N 8

extern __device__ int g[N];

extern __device__ void bar(void);
```

```c++
//---------- b.cu ----------
#include "b.h"

__device__ int g[N];

__device__ void bar (void)
{
  g[threadIdx.x]++;
}
```

```c++
//---------- a.cu ----------
#include <stdio.h>
#include "b.h"

__global__ void foo (void) {

  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;

  __syncthreads();

  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];

  bar();
}

int main (void) {
  unsigned int i;
  int *dg, hg[N];
  int sum = 0;

  foo<<<1, N>>>();

  if(cudaGetSymbolAddress((void**)&dg, g)){
      printf("couldn't get the symbol addr\n");
      return 1;
  }
  if(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost)){
      printf("couldn't memcpy\n");
      return 1;
  }

  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }

  return 0;
}
```

These can be compiled with the following commands (these examples are for Linux):

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 a.o b.o
```

If you want to invoke the device and host linker separately, you can do:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 --device-link a.o b.o --output-file link.o
g++ a.o b.o link.o --library-path=<path> --library=cudart
```

Note that all desired target architectures must be passed to the device linker, as that specifies what will be in the final executable (some objects or libraries may contain device code for multiple architectures, and the link step can then choose what to put in the final executable).

If you want to use the driver API to load a linked cubin, you can request just the cubin:

```text
nvcc --gpu-architecture=sm_100 --device-link a.o b.o \
    --cubin --output-file link.cubin
```

The objects could be put into a library and used with:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --lib a.o b.o --output-file test.a
nvcc --gpu-architecture=sm_100 test.a
```

Note that only static libraries are supported by the device linker.

A PTX file can be compiled to a host object file and then linked by using:

```text
nvcc --gpu-architecture=sm_100 --device-c a.ptx
```

An example that uses libraries, host linker, and dynamic parallelism would be:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 --device-link a.o b.o --output-file link.o
nvcc --lib --output-file libgpu.a a.o b.o link.o
g++ host.o --library=gpu --library-path=<path> \
    --library=cudadevrt --library=cudart
```

It is possible to do multiple device links within a single host executable, as long as each device link is independent of the other. This requirement of independence means that they cannot share code across device executables, nor can they share addresses (e.g., a device function address can be passed from host to device for a callback only if the device link sees both the caller and potential callback callee; you cannot pass an address from one device executable to another, as those are separate address spaces).

## [6.5. Optimization Of Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#optimization-of-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optimization-of-separate-compilation "Permalink to this headline")

Separately compiled code may not have as high of performance as whole program code because of the inability to inline code across files. A way to still get optimal performance is to use link-time optimization, which stores intermediate code which is then linked together to perform high level optimizations. This can be done with the `--dlink-time-opt` or `-dlto` option. This option must be specified at both compile and link time. If only some of the files are compiled with `-dlto`, then those will be linked and optimized together while the rest uses the normal separate compilation. A side effect is that this shifts some of the compile time to the link phase, and there may be some scalability issues with really large codes. If you want to compile using `-gencode` to build for multiple arch, use `-dc -gencode arch=compute_NN,code=lto_NN` to specify the intermediate IR to be stored (where `NN` is the SM architecture version). Then use `-dlto` option to link for a specific architecture.

As of CUDA 12.0 there is support for runtime LTO via the `nvJitLink` library.

## [6.6. Potential Separate Compilation Issues](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#potential-separate-compilation-issues)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#potential-separate-compilation-issues "Permalink to this headline")

### [6.6.1. Object Compatibility](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#object-compatibility)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#object-compatibility "Permalink to this headline")

Only relocatable device code with the same ABI version, link-compatible SM target architecture, and same pointer size (32 or 64) can be linked together. The toolkit version of the linker must be >= the toolkit version of the objects. Incompatible objects will produce a link error. Link-compatible SM architectures are ones that have compatible SASS binaries that can combine without translating, e.g. sm_86 and sm_80. An object could have been compiled for a different architecture but also have PTX available, in which case the device linker will JIT the PTX to cubin for the desired architecture and then link.

If Link Time Optimization is used with `-dlto`, the intermediate LTOIR is only guaranteed to be compatible within a major release (e.g. can link together 12.0 and 12.1 LTO intermediates, but not 12.1 and 11.6).

If a kernel is limited to a certain number of registers with the `launch_bounds` attribute or the `--maxrregcount` option, then all functions that the kernel calls must not use more than that number of registers; if they exceed the limit, then a link error will be given.

### [6.6.2. JIT Linking Support](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#jit-linking-support)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#jit-linking-support "Permalink to this headline")

JIT linking means doing an implicit relink of the code at load time. If the cubin does not match the target architecture at load time, the driver re-invokes the device linker to generate cubin for the target architecture, by first JIT’ing the PTX for each object to the appropriate cubin, and then linking together the new cubin. If PTX or cubin for the target architecture is not found for an object, then the link will fail. Implicit JIT linking of the LTO intermediates is not supported at this time, although they can be explicitly linked with the `nvJitLink` library.

### [6.6.3. Implicit CUDA Host Code](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#implicit-cuda-host-code)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#implicit-cuda-host-code "Permalink to this headline")

A file like `b.cu` above only contains CUDA device code, so one might think that the b.o object doesn’t need to be passed to the host linker. But actually there is implicit host code generated whenever a device symbol can be accessed from the host side, either via a launch or an API call like `cudaGetSymbolAddress()`. This implicit host code is put into `b.o`, and needs to be passed to the host linker. Plus, for JIT linking to work all device code must be passed to the host linker, else the host executable will not contain device code needed for the JIT link. So a general rule is that the device linker and host linker must see the same host object files (if the object files have any device references in them—if a file is pure host then the device linker doesn’t need to see it). If an object file containing device code is not passed to the host linker, then you will see an error message about the function `__cudaRegisterLinkedBinary_name` calling an undefined or unresolved symbol `__fatbinwrap_name`.

### [6.6.4. Using __CUDA_ARCH__](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#using-cuda-arch)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-cuda-arch "Permalink to this headline")

In separate compilation, `__CUDA_ARCH__` must not be used in headers such that different objects could contain different behavior. Or, it must be guaranteed that all objects will compile for the same compute_arch. If a weak function or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then the instances of that function in the objects could conflict if the objects are compiled for different compute arch. For example, if an a.h contains:

```c++
template<typename T>
__device__ T* getptr(void)
{
#if __CUDA_ARCH__ == 800
  return NULL; /* no address */
#else
  __shared__ T arr[256];
  return arr;
#endif
}
```

Then if a.cu and b.cu both include a.h and instantiate `getptr` for the same type, and b.cu expects a non-NULL address, and compile with:

```text
nvcc --gpu-architecture=compute_80 --device-c a.cu
nvcc --gpu-architecture=compute_86 --device-c b.cu
nvcc --gpu-architecture=sm_86 a.o b.o
```

At link time only one version of the getptr is used, so the behavior would depend on which version is picked. To avoid this, either a.cu and b.cu must be compiled for the same compute arch, or `__CUDA_ARCH__` should not be used in the shared header function.

### [6.6.5. Device Code in Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#device-code-in-libraries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#device-code-in-libraries "Permalink to this headline")

If a device function with non-weak external linkage is defined in a library as well as a non-library object (or another library), the device linker will complain about the multiple definitions (this differs from traditional host linkers that may ignore the function definition from the library object, if it was already found in an earlier object).
