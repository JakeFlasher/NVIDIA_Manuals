---
title: "4.9.2. A Barrier’s Phase: Arrival, Countdown, Completion, and Reset"
section: "4.9.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#a-barrier-s-phase-arrival-countdown-completion-and-reset"
---

## [4.9.2. A Barrier’s Phase: Arrival, Countdown, Completion, and Reset](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#a-barrier-s-phase-arrival-countdown-completion-and-reset)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#a-barrier-s-phase-arrival-countdown-completion-and-reset "Permalink to this headline")

An asynchronous barrier counts down from the expected arrival count to zero as participating threads call `bar.arrive()`. When the countdown reaches zero, the barrier is complete for the current phase. When the last call to `bar.arrive()` causes the countdown to reach zero, the countdown is automatically and atomically reset. The reset assigns the countdown to the expected arrival count, and moves the barrier to the next phase.

A `token` object of class `cuda::barrier::arrival_token`, as returned from `token=bar.arrive()`, is associated with the current phase of the barrier. A call to `bar.wait(std::move(token))` blocks the calling thread while the barrier is in the current phase, i.e., while the phase associated with the token matches the phase of the barrier. If the phase is advanced (because the countdown reaches zero) before the call to `bar.wait(std::move(token))` then the thread does not block; if the phase is advanced while the thread is blocked in `bar.wait(std::move(token))`, the thread is unblocked.

**It is essential to know when a reset could or could not occur, especially in non-trivial arrive/wait synchronization patterns.**

- A thread’s calls to `token=bar.arrive()` and `bar.wait(std::move(token))` must be sequenced such that `token=bar.arrive()` occurs during the barrier’s current phase, and `bar.wait(std::move(token))` occurs during the same or next phase.
- A thread’s call to `bar.arrive()` must occur when the barrier’s counter is non-zero. After barrier initialization, if a thread’s call to `bar.arrive()` causes the countdown to reach zero then a call to `bar.wait(std::move(token))` must happen before the barrier can be reused for a subsequent call to `bar.arrive()`.
- `bar.wait()` must only be called using a `token` object of the current phase or the immediately preceding phase. For any other values of the `token` object, the behavior is undefined.

For simple arrive/wait synchronization patterns, compliance with these usage rules is straightforward.
