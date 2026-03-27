---
title: "4.7.4.2. Checking if Lazy Loading is Enabled at Runtime"
section: "4.7.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#checking-if-lazy-loading-is-enabled-at-runtime"
---

### [4.7.4.2. Checking if Lazy Loading is Enabled at Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#checking-if-lazy-loading-is-enabled-at-runtime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#checking-if-lazy-loading-is-enabled-at-runtime "Permalink to this headline")

The `cuModuleGetLoadingMode` API in the CUDA driver API can be used to determine if lazy loading is enabled. Note that CUDA must be initialized before running this function. Sample usage is shown in the snippet below.

```c++
#include "<cuda.h>"
#include "<assert.h>"
#include "<iostream>"

int main() {
        CUmoduleLoadingMode mode;

        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}
```
