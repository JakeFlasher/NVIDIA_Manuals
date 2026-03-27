---
title: "5.4.8.3. Low-Level Load and Store Functions"
section: "5.4.8.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#low-level-load-and-store-functions"
---

### [5.4.8.3. Low-Level Load and Store Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#low-level-load-and-store-functions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#low-level-load-and-store-functions "Permalink to this headline")

```cuda
T __ldg(const T* address);
```

The function `__ldg()` performs a read-only L1/Tex cache load. It supports all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```cuda
T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);
```

The functions perform a load using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```cuda
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```

The functions perform a store using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.
