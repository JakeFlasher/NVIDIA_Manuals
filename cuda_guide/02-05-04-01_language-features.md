---
title: "2.5.4.1. Language Features"
section: "2.5.4.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#language-features"
---

### [2.5.4.1. Language Features](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#language-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#language-features "Permalink to this headline")

`nvcc` supports the C++ core language features, from C++03 to [C++20](https://en.cppreference.com/w/cpp/compiler_support#cpp20). The `-std` flag can be used to specify the language standard to use:

- `--std={c++03|c++11|c++14|c++17|c++20}`

In addition, `nvcc` supports the following language extensions:

- `-restrict`: Assert that all kernel pointer parameters are [restrict](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#restrict) pointers.
- `-extended-lambda`: Allow `__host__`, `__device__` annotations in lambda declarations.
- `-expt-relaxed-constexpr`: (Experimental flag) Allow host code to invoke `__device__ constexpr` functions, and device code to invoke `__host__ constexpr` functions.

More detail on these features can be found in the [extended lambda](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambdas) and [constexpr](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#constexpr-functions) sections.
