---
title: "5.4.7.2. __CUDA_ARCH_SPECIFIC__ and __CUDA_ARCH_FAMILY_SPECIFIC__"
section: "5.4.7.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#cuda-arch-specific-and-cuda-arch-family-specific"
---

### [5.4.7.2. __CUDA_ARCH_SPECIFIC__ and __CUDA_ARCH_FAMILY_SPECIFIC__](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-arch-specific-and-cuda-arch-family-specific)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-arch-specific-and-cuda-arch-family-specific "Permalink to this headline")

The macros `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are defined to identify GPU devices with [architecture-](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-architecture-specific-features)  and [family-](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-family-specific-features) specific features, respectively. See [Feature Set Compiler Targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-feature-set-compiler-targets) section for more information.

Similarly to `__CUDA_ARCH__`, `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are only defined in the device code, namely in the `__device__`, `__host__ __device__`, and `__global__` functions. The macros are associated with the `nvcc` options  `compute_<version>a` and `compute_<version>f`.

```bash
nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu
```

- `__CUDA_ARCH__ == 1000`.
- `__CUDA_ARCH_SPECIFIC__ == 1000`.
- `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.

```bash
nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu
```

- `__CUDA_ARCH__ == 1000`.
- `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.
- `__CUDA_ARCH_SPECIFIC__` is not defined.

```bash
nvcc -arch=sm_100 prog.cu
```

- `__CUDA_ARCH__ == 1000`.
- `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.
- `__CUDA_ARCH_SPECIFIC__` is not defined.

```bash
nvcc -arch=sm_100a prog.cu
# equivalent to:
nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu
```

- `__CUDA_ARCH__ == 1000`.
- `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.
- `__CUDA_ARCH_SPECIFIC__ == 1000` and `__CUDA_ARCH_SPECIFIC__` not defined are both generated.
