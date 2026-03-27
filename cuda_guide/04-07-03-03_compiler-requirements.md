---
title: "4.7.3.3. Compiler Requirements"
section: "4.7.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#compiler-requirements"
---

### [4.7.3.3. Compiler Requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#compiler-requirements)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#compiler-requirements "Permalink to this headline")

Lazy loading does not require any compiler support. Both SASS and PTX compiled with pre-11.7 compilers can be loaded with lazy loading enabled, and will see full benefits of the feature. However, the version 11.7+ CUDA runtime is still required, as described above.
