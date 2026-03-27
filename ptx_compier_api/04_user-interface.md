---
title: "4. User Interface"
section: "4"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#user-interface"
---

# [4. User Interface](https://docs.nvidia.com/cuda/ptx-compiler-api#user-interface)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#user-interface "Permalink to this headline")

This chapter presents the PTX Compiler APIs. Basic usage of the API is explained in Basic Usage.

- [PTX-compiler handle](https://docs.nvidia.com/cuda/ptx-compiler-api/#ptx-compiler-handle)
- [Error codes](https://docs.nvidia.com/cuda/ptx-compiler-api/#error-codes)
- [API Versioning](https://docs.nvidia.com/cuda/ptx-compiler-api/#api-versioning)
- [Compilation APIs](https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-apis)

## [4.1. PTX-Compiler Handle](https://docs.nvidia.com/cuda/ptx-compiler-api#ptx-compiler-handle)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#ptx-compiler-handle "Permalink to this headline")

**Typedefs**

**[nvPTXCompilerHandle](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__handle_1ga78e3bba9412ef3e505a9a908089c883e)**

  nvPTXCompilerHandle represents a handle to the PTX Compiler.

### [4.1.1. Typedefs](https://docs.nvidia.com/cuda/ptx-compiler-api#typedefs)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#typedefs "Permalink to this headline")

```
`typedef` `struct` `nvPTXCompiler` `*``nvPTXCompilerHandle`
```

nvPTXCompilerHandle represents a handle to the PTX Compiler.

To compile a PTX program string, an instance of nvPTXCompiler must be created and the handle to it must be obtained using the API [nvPTXCompilerCreate()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga86b0c1159bfd34f0fdf632101f30ca26). Then the compilation can be done using the API [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38).

## [4.2. Error codes](https://docs.nvidia.com/cuda/ptx-compiler-api#error-codes)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#error-codes "Permalink to this headline")

**Enumerations**

**[nvPTXCompileResult](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)**

  The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result.

### [4.2.1. Enumerations](https://docs.nvidia.com/cuda/ptx-compiler-api#enumerations)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#enumerations "Permalink to this headline")

```
`enum` `nvPTXCompileResult`
```

The nvPTXCompiler APIs return the nvPTXCompileResult codes to indicate the call result.

_Values:_

```
`enumerator` `NVPTXCOMPILE_SUCCESS`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INVALID_INPUT`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_COMPILATION_FAILURE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_INTERNAL`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_OUT_OF_MEMORY`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC`
```

```
`enumerator` `NVPTXCOMPILE_ERROR_CANCELLED`
```

```
`enumerator` `NVPTXCOMPILE_PARSE_ONLY_SUCCESS`
```

## [4.3. API Versioning](https://docs.nvidia.com/cuda/ptx-compiler-api#api-versioning)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#api-versioning "Permalink to this headline")

The PTX compiler APIs are versioned so that any new features or API changes can be done by bumping up the API version.

**Functions**

**nvPTXCompileResult [nvPTXCompilerGetNumSupportedArchs](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga5bc657a75e33cad967df8da7d1387511)(int *numArchs)**

  Sets the output paramater `numArchs` with the number of architectures supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetNumSupportedVersions](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1gae6aa43abe2b0ab4bd948c58984cf44ae)(int *numVersions)**

  Sets the output paramater `numVersions` with the number of PTX ISA versions supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetSupportedArchs](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga0abce69c5235af0da34fb09171e46c2a)(int *supportedArchs)**

  Populates the array passed via output paramater `supportedArchs` with architectures supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetSupportedVersions](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1ga916f37c92d682ef18c552e9e767c465a)(int(*versions)[2])**

  Populates the 2D Array passed via output paramater `versions` with PTX ISA Versions major, minor pair supported by PTX Compiler Library.

**nvPTXCompileResult [nvPTXCompilerGetVersion](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__versioning_1gadb846da0f180ce336bf1f050ce5b3e6b)(unsigned int *major, unsigned int *minor)**

  Queries the current `major` and `minor` version of PTX Compiler APIs being used.

### [4.3.1. Functions](https://docs.nvidia.com/cuda/ptx-compiler-api#functions)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#functions "Permalink to this headline")

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetNumSupportedArchs`(`int` `*``numArchs`)
```

Sets the output paramater `numArchs` with the number of architectures supported by PTX Compiler Library.

This can be used to pass an array to nvPTXCompilerGetSupportedArchs to get the supported architectures.

**Parameters**
: **numArchs** – **[out]** Number of supported architectures

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetNumSupportedVersions`(`int` `*``numVersions`)
```

Sets the output paramater `numVersions` with the number of PTX ISA versions supported by PTX Compiler Library.

This can be used to pass a 2D Array to nvPTXCompilerGetSupportedVersions. Number of rows in 2D Array will be equal to value populated in `numVersions`.

**Parameters**
: **numVersions** – **[out]** Number of supported PTX ISA versions

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetSupportedArchs`(`int` `*``supportedArchs`)
```

Populates the array passed via output paramater `supportedArchs` with architectures supported by PTX Compiler Library.

The array is sorted in the ascending order. The size of the array to be passed can be determined using nvPTXCompilerGetNumSupportedArchs.

**Parameters**
: **supportedArchs** – **[out]** Sorted array of supported architectures.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetSupportedVersions`(`int` `(``*``versions``)``[``2``]`)
```

Populates the 2D Array passed via output paramater `versions` with PTX ISA Versions major, minor pair supported by PTX Compiler Library.

The array is sorted in the ascending order. the row size of the 2D array can be determined using nvPTXCompilerGetNumSupportedVersions.

**Parameters**
: **versions** – **[out]** Sorted 2D array with column size 2, having pair of major and minor PTX ISA version supported.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetVersion`(`unsigned` `int` `*``major`, `unsigned` `int` `*``minor`)
```

Queries the current `major` and `minor` version of PTX Compiler APIs being used.

> **Note**
>
> The version of PTX Compiler APIs follows the CUDA Toolkit versioning. The PTX ISA version supported by a PTX Compiler API version is listed [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes).

**Parameters**
: - **major** – **[out]** Major version of the PTX Compiler APIs
- **minor** – **[out]** Minor version of the PTX Compiler APIs

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

## [4.4. Compilation APIs](https://docs.nvidia.com/cuda/ptx-compiler-api#compilation-apis)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#compilation-apis "Permalink to this headline")

**Functions**

**nvPTXCompileResult [nvPTXCompilerCompile](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38)(nvPTXCompilerHandle compiler, int numCompileOptions, const char *const *compileOptions)**

  Compile a PTX program with the given compiler options.

**nvPTXCompileResult [nvPTXCompilerCreate](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga86b0c1159bfd34f0fdf632101f30ca26)(nvPTXCompilerHandle *compiler, size_t ptxCodeLen, const char *ptxCode)**

  Obtains the handle to an instance of the PTX compiler initialized with the given PTX program `ptxCode` .

**nvPTXCompileResult [nvPTXCompilerDestroy](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gabe1fd08fafc16166b553aa1f2ac132f4)(nvPTXCompilerHandle *compiler)**

  Destroys and cleans the already created PTX compiler.

**nvPTXCompileResult [nvPTXCompilerGetCompiledProgram](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga8f56c9115a219cb518d15ea8e21cd633)(nvPTXCompilerHandle compiler, void *binaryImage)**

  Obtains the image of the compiled program.

**nvPTXCompileResult [nvPTXCompilerGetCompiledProgramSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga026bf92499c9c0e362b4b899b1550f6a)(nvPTXCompilerHandle compiler, size_t *binaryImageSize)**

  Obtains the size of the image of the compiled program.

**nvPTXCompileResult [nvPTXCompilerGetErrorLog](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gaaf88564e76bf5314a597296ba1397942)(nvPTXCompilerHandle compiler, char *errorLog)**

  Query the error message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetErrorLogSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga197d976f96d939b6da998bd10c8b591f)(nvPTXCompilerHandle compiler, size_t *errorLogSize)**

  Query the size of the error message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetInfoLog](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gae35e6f03a99742a9c008c1a03b6dc35f)(nvPTXCompilerHandle compiler, char *infoLog)**

  Query the information message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerGetInfoLogSize](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gabff992a534a0d8fde596bfbe8da08449)(nvPTXCompilerHandle compiler, size_t *infoLogSize)**

  Query the size of the information message that was seen previously for the handle.

**nvPTXCompileResult [nvPTXCompilerSetFlowCallback](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1gae9e9942cf8b8363353cea54a9277e667)(nvPTXCompilerHandle compiler, int(*callback)(void *, void *), void *payload)**

  Register a callback function that the compiler will invoke at different phases of PTX Compilation during a call to [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) .

### [4.4.1. Functions](https://docs.nvidia.com/cuda/ptx-compiler-api#id1)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#id1 "Permalink to this headline")

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerCompile`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `int` `numCompileOptions`, `const` `char` `*``const` `*``compileOptions`)
```

Compile a PTX program with the given compiler options.

> **Note**
>
> &#8212;gpu-name (-arch) is a mandatory option.

**Parameters**
: - **compiler** – **[inout]** A handle to PTX compiler initialized with the PTX program which is to be compiled. The compiled program can be accessed using the handle
- **numCompileOptions** – **[in]** Length of the array `compileOptions`
- **compileOptions** – **[in]** Compiler options with which compilation should be done. The compiler options string is a null terminated character array. A valid list of compiler options is at [link](http://docs.nvidia.com/cuda/ptx-compiler-api/index.html#compile-options).

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILATION_FAILURE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerCreate`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `*``compiler`, `size_t` `ptxCodeLen`, `const` `char` `*``ptxCode`)
```

Obtains the handle to an instance of the PTX compiler initialized with the given PTX program `ptxCode`.

**Parameters**
: - **compiler** – **[out]** Returns a handle to PTX compiler initialized with the PTX program `ptxCode`
- **ptxCodeLen** – **[in]** Size of the PTX program `ptxCode` passed as string
- **ptxCode** – **[in]** The PTX program which is to be compiled passed as string.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerDestroy`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `*``compiler`)
```

Destroys and cleans the already created PTX compiler.

**Parameters**
: **compiler** – **[in]** A handle to the PTX compiler which is to be destroyed

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_OUT_OF_MEMORY](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetCompiledProgram`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `void` `*``binaryImage`)
```

Obtains the image of the compiled program.

> **Note**
>
> [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) API should be invoked for the handle before calling this API. Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **binaryImage** – **[out]** The image of the compiled program. Client should allocate memory for `binaryImage`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetCompiledProgramSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``binaryImageSize`)
```

Obtains the size of the image of the compiled program.

> **Note**
>
> [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) API should be invoked for the handle before calling this API. Otherwise, NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE is returned.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **binaryImageSize** – **[out]** The size of the image of the compiled program

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetErrorLog`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `char` `*``errorLog`)
```

Query the error message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **errorLog** – **[out]** The error log which was produced in previous call to nvPTXCompilerCompiler(). Clients should allocate memory for `errorLog`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetErrorLogSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``errorLogSize`)
```

Query the size of the error message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **errorLogSize** – **[out]** The size of the error log in bytes which was produced in previous call to nvPTXCompilerCompiler().

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetInfoLog`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `char` `*``infoLog`)
```

Query the information message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **infoLog** – **[out]** The information log which was produced in previous call to nvPTXCompilerCompiler(). Clients should allocate memory for `infoLog`

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerGetInfoLogSize`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `size_t` `*``infoLogSize`)
```

Query the size of the information message that was seen previously for the handle.

**Parameters**
: - **compiler** – **[in]** A handle to PTX compiler on which [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38) has been performed.
- **infoLogSize** – **[out]** The size of the information log in bytes which was produced in previous call to nvPTXCompilerCompiler().

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INTERNAL](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)

```
[`nvPTXCompileResult`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv418nvPTXCompileResult "nvPTXCompileResult") `nvPTXCompilerSetFlowCallback`([`nvPTXCompilerHandle`](https://docs.nvidia.com/cuda/ptx-compiler-api/#_CPPv419nvPTXCompilerHandle "nvPTXCompilerHandle") `compiler`, `int` `(``*``callback``)``(``void``*``,` `void``*``)`, `void` `*``payload`)
```

Register a callback function that the compiler will invoke at different phases of PTX Compilation during a call to [nvPTXCompilerCompile()](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__compilation_1ga3d08161ba6e6a5fb40814900ce0f2e38).

The callback function decides to cancel the compilation by returning specific values.

Callback function must satisfy the following constraints

(1) Its signature should be

```c++
int callback(void* param1, void* param2);
```

 When invoking the callback, the compiler will always pass `payload` to param1 so that the callback may make decisions based on `payload` . It’ll always pass NULL to param2 for now which is reserved for future extensions.
(2) It must return 1 to cancel compilation or 0 to continue. Other return values are reserved for future use.

(3) It must return consistent values. Once it returns 1 at one point, it must return 1 in all following invocations during the current nvPTXCompilerCompile call in progress.

(4) It must be thread-safe.

(5) It must not invoke any nvrtc/libnvvm/ptx APIs.

**Parameters**
: - **compiler** – **[in]** A handle to an initialized PTX compiler in which to introduce the callback.
- **callback** – **[in]** Function pointer to the callback function.
- **payload** – **[in]** payload to be passed as a parameter when invoking the callback.

**Returns**
: - [NVPTXCOMPILE_SUCCESS](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01) [NVPTXCOMPILE_ERROR_INVALID_INPUT](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
- [NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE](https://docs.nvidia.com/cuda/ptx-compiler-api/#group__error_1ga402c9248a2b7f74f2d56fa8c292f4e01)
