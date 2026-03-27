---
title: "5.8. CUDA C++ Execution model"
section: "5.8"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html#cuda-c-execution-model"
---

# [5.8. CUDA C++ Execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#cuda-c-execution-model)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cuda-c-execution-model "Permalink to this headline")

CUDA C++ aims to provide [parallel forward progress [intro.progress.9]](https://eel.is/c++draft/intro.progress#9)
for all device threads of execution, facilitating the parallelization of pre-existing C++ applications with CUDA C++.

<details>
<summary>[[intro.progress]](https://eel.is/c++draft/intro.progress)</summary>

- [[intro.progress.7]](https://eel.is/c++draft/intro.progress#7): For a thread of execution
providing [concurrent forward progress guarantees](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees),
the implementation ensures that the thread will eventually make progress for as long as it has not terminated.

[Note 5: This applies regardless of whether or not other threads of execution (if any) have been or are making progress.
To eventually fulfill this requirement means that this will happen in an unspecified but finite amount of time. — end note]
- [[intro.progress.9]](https://eel.is/c++draft/intro.progress#9): For a thread of execution providing
[parallel forward progress guarantees](https://eel.is/c++draft/intro.progress#9), the implementation is not required to ensure that
the thread will eventually make progress if it has not yet executed any execution step; once this thread has executed a step,
it provides [concurrent forward progress guarantees](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees).

> [Note 6: This does not specify a requirement for when to start this thread of execution, which will typically be specified by the entity
> that creates this thread of execution. For example, a thread of execution that provides concurrent forward progress guarantees and executes
> tasks from a set of tasks in an arbitrary order, one after the other, satisfies the requirements of parallel forward progress for these
> tasks. — end note]

</details>

The CUDA C++ Programming Language is an extension of the C++ Programming Language.
This section documents the modifications and extensions to the [[intro.progress]](https://eel.is/c++draft/intro.progress) section of the current [ISO International Standard ISO/IEC 14882 – Programming Language C++](https://eel.is/c++draft/) draft.
Modified sections are called out explicitly and their diff is shown in **bold**.
All other sections are additions.
