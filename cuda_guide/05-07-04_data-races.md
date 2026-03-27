---
title: "5.7.4. Data Races"
section: "5.7.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#data-races"
---

## [5.7.4. Data Races](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#data-races)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#data-races "Permalink to this headline")

Modify [intro.races paragraph 21](https://eel.is/c++draft/intro.races#21) of ISO/IEC IS 14882 (the C++ Standard)
as follows:

> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at
> least one of which is not atomic **at a scope that includes the thread that performed the other operation**, and neither
> happens before the other, except for the special case for signal handlers described below.
> Any such data race results in undefined behavior. […]

Modify [thread.barrier.class paragraph 4](https://eel.is/c++draft/thread.barrier.class#4) of ISO/IEC IS
14882 (the C++ Standard) as follows

> 4. Concurrent invocations of the member functions of `barrier`, other than its destructor, do not introduce data
> races **as if they were atomic operations**. […]

Modify [thread.latch.class paragraph 2](https://eel.is/c++draft/thread.latch.class#2) of ISO/IEC IS 14882
(the C++ Standard) as follows:

> 2. Concurrent invocations of the member functions of `latch`, other than its destructor, do not introduce data
> races **as if they were atomic operations**. […]

Modify [thread.sema.cnt paragraph 3](https://eel.is/c++draft/thread.sema.cnt#3) of ISO/IEC IS 14882
(the C++ Standard) as follows:

> 3. Concurrent invocations of the member functions of `counting_semaphore`, other than its destructor, do not
> introduce data races **as if they were atomic operations**.

Modify [thread.stoptoken.intro paragraph 5](https://eel.is/c++draft/thread#stoptoken.intro-5) of ISO/IEC IS
14882 (the C++ Standard) as follows:

> Calls to the functions `request_stop`, `stop_requested`, and `stop_possible` do not introduce data
> races **as if they were atomic operations**. […]

Modify [atomics.fences paragraph 2 through 4](https://eel.is/c++draft/atomics.fences#2) of ISO/IEC IS 14882 (the
C++ Standard) as follows:

> A release fence A synchronizes with an acquire fence B if there exist atomic operations X and Y, both operating on
> some atomic object M, such that A is sequenced before X, X modifies M, Y is sequenced before B, and Y reads the value
> written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a
> release operation, **and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed
> each other operation**.
>
>
> A release fence A synchronizes with an atomic operation B that performs an acquire operation on an atomic object M if
> there exists an atomic operation X such that A is sequenced before X, X modifies M, and B reads the value written by
> X or a value written by any side effect in the hypothetical release sequence X would head if it were a release
> operation, **and each operation (A, B, and X) specifies a scope that includes the thread that performed each other
> operation**.
>
>
> An atomic operation A that is a release operation on an atomic object M synchronizes with an acquire fence B if
> there exists some atomic operation X on M such that X is sequenced before B and reads the value written by A or a
> value written by any side effect in the release sequence headed by A, **and each operation (A, B, and X) specifies
> a scope that includes the thread that performed each other operation**.
