---
title: "5.8.1. Host threads"
section: "5.8.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html#host-threads"
---

## [5.8.1. Host threads](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#host-threads)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#host-threads "Permalink to this headline")

The forward progress provided by threads of execution created by the host implementation to
execute [main](https://en.cppreference.com/w/cpp/language/main_function), [std::thread](https://en.cppreference.com/w/cpp/thread/thread),
and [std::jthread](https://en.cppreference.com/w/cpp/thread/jthread) is implementation-defined behavior of the host
implementation [[intro.progress]](https://eel.is/c++draft/intro.progress).
General-purpose host implementations should provide concurrent forward progress.

If the host implementation provides [concurrent forward progress [intro.progress.7]](https://eel.is/c++draft/intro.progress#7),
then CUDA C++ provides [parallel forward progress [intro.progress.9]](https://eel.is/c++draft/intro.progress#9) for device threads.
