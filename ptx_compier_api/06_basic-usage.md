---
title: "6. Basic Usage"
section: "6"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#basic-usage"
---

# [6. Basic Usage](https://docs.nvidia.com/cuda/ptx-compiler-api#basic-usage)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#basic-usage "Permalink to this headline")

This section of the document uses a simple example, _Vector Addition_, shown in [Figure 1](https://docs.nvidia.com/cuda/ptx-compiler-api/#ptx-source-string-for-simple-vector-addition) to explain how to use PTX Compiler
APIs to compile this PTX program. For brevity and readability, error checks on the API return values
are not shown.

_Figure 1. PTX source string for a simple vector addition_

```text
const char *ptxCode = "                                    \n \
   .version 7.0                                            \n \
   .target sm_50                                           \n \
   .address_size 64                                        \n \
   .visible .entry simpleVectorAdd(                        \n \
        .param .u64 simpleVectorAdd_param_0,               \n \
        .param .u64 simpleVectorAdd_param_1,               \n \
        .param .u64 simpleVectorAdd_param_2                \n \
   ) {                                                     \n \
        .reg .f32   %f<4>;                                 \n \
        .reg .b32   %r<5>;                                 \n \
        .reg .b64   %rd<11>;                               \n \
        ld.param.u64    %rd1, [simpleVectorAdd_param_0];   \n \
        ld.param.u64    %rd2, [simpleVectorAdd_param_1];   \n \
        ld.param.u64    %rd3, [simpleVectorAdd_param_2];   \n \
        cvta.to.global.u64      %rd4, %rd3;                \n \
        cvta.to.global.u64      %rd5, %rd2;                \n \
        cvta.to.global.u64      %rd6, %rd1;                \n \
        mov.u32         %r1, %ctaid.x;                     \n \
        mov.u32         %r2, %ntid.x;                      \n \
        mov.u32         %r3, %tid.x;                       \n \
        mad.lo.s32      %r4, %r2, %r1, %r3;                \n \
        mul.wide.u32    %rd7, %r4, 4;                      \n \
        add.s64         %rd8, %rd6, %rd7;                  \n \
        ld.global.f32   %f1, [%rd8];                       \n \
        add.s64         %rd9, %rd5, %rd7;                  \n \
        ld.global.f32   %f2, [%rd9];                       \n \
        add.f32         %f3, %f1, %f2;                     \n \
        add.s64         %rd10, %rd4, %rd7;                 \n \
        st.global.f32   [%rd10], %f3;                      \n \
        ret;                                               \n \
   } ";
```

The CUDA code corresponding to this PTX program would look like:

_Figure 2. Equivalent CUDA source for the simple vector addition_

```text
extern "C"
 __global__ void simpleVectorAdd(float *x, float *y, float *out)
 {
     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
     out[tid] = x[tid] + y[tid];
 }
```

With this PTX program as a string, we can create the compiler and obtain a handle to it as shown in
[Figure 3](https://docs.nvidia.com/cuda/ptx-compiler-api/#compiler-creation-and-initialization-of-a-program).

_Figure 3. Compiler creation and initialization of a program_

```text
nvPTXCompilerHandle compiler;
nvPTXCompilerCreate(&compiler, (size_t)strlen(ptxCode), ptxCode);
```

Compilation can now be done by specifying the compile options as shown in [Figure 4](https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-of-ptx-program).

_Figure 4. Compilation of the PTX program_

```text
const char* compile_options[] = { "--gpu-name=sm_80",
                                  "--verbose"
                                };

nvPTXCompilerCompile(compiler, 2, compile_options);
```

The compiled GPU assembly code can now be obtained. To obtain this we first allocate memory for
it. And to allocate memory, we need to query the size of the image of the compiled GPU assembly code
which is done as shown in [Figure 5](https://docs.nvidia.com/cuda/ptx-compiler-api/#query-size-of-compiled-assembly-image).

_Figure 5. Query size of the compiled assembly image_

```text
nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize);
```

The image of the compiled GPU assembly code can now be queried as shown in [Figure 6](https://docs.nvidia.com/cuda/ptx-compiler-api/#query-the-compiled-assembly-image). This image can then be executed on the GPU by passing this
image to the CUDA Driver APIs.

_Figure 6. Query the compiled assembly image_

```text
elf = (char*) malloc(elfSize);
nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
```

When the compiler is not needed anymore, it can be destroyed as shown in [Figure 7](https://docs.nvidia.com/cuda/ptx-compiler-api/#destroy-the-compiler).

_Figure 7. Destroy the compiler_

```text
nvPTXCompilerDestroy(&compiler);
```
