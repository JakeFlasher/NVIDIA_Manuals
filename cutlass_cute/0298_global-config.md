---
title: "Global Config"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html#global-config"
---

### [Global Config](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#global-config)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#global-config "Permalink to this headline")

Here is one example for a global config.
On linux this is usually located at `~/.config/clangd/config.yaml` . Here is one example config for CUDA projects on SM90.
The key settings here are the preprocessor vars (`-D__CUDACC_VER_MAJOR__` , `-D__CUDA_ARCH__`)

```console
CompileFlags:
  Compiler: /usr/local/cuda/bin/nvcc
  Add:
    - --cuda-path=/usr/local/cuda
    - --cuda-gpu-arch=sm_90a
    - -I/usr/local/cuda/include
    - "-xcuda"
    # report all errors
    - "-ferror-limit=0"
    - --cuda-gpu-arch=sm_90a
    - --std=c++17
    - "-D__INTELLISENSE__"
    - "-D__CLANGD__"
    - "-DCUDA_12_0_SM90_FEATURES_SUPPORTED"
    - "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1"
    - "-D_LIBCUDACXX_STD_VER=12"
    - "-D__CUDACC_VER_MAJOR__=12"
    - "-D__CUDACC_VER_MINOR__=3"
    - "-D__CUDA_ARCH__=900"
    - "-D__CUDA_ARCH_FEAT_SM90_ALL"
    - "-Wno-invalid-constexpr"
  Remove:
    # strip CUDA fatbin args
    - "-Xfatbin*"
    # strip CUDA arch flags
    - "-gencode*"
    - "--generate-code*"
    # strip CUDA flags unknown to clang
    - "-ccbin*"
    - "--compiler-options*"
    - "--expt-extended-lambda"
    - "--expt-relaxed-constexpr"
    - "-forward-unknown-to-host-compiler"
    - "-Werror=cross-execution-space-call"
Hover:
  ShowAKA: No
InlayHints:
  Enabled: No
Diagnostics:
  Suppress:
    - "variadic_device_fn"
    - "attributes_not_allowed"
```
