---
title: "4.9.7. Producer-Consumer Pattern Using Barriers"
section: "4.9.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#producer-consumer-pattern-using-barriers"
---

## [4.9.7. Producer-Consumer Pattern Using Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#producer-consumer-pattern-using-barriers)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#producer-consumer-pattern-using-barriers "Permalink to this headline")

A thread block can be spatially partitioned to allow different threads to perform independent operations. This is most commonly done by assigning threads from different warps within the thread block to specific tasks. This technique is referred to as _warp specialization_.

This section shows an example of spatial partitioning in a producer-consumer pattern, where one subset of threads produces data that is concurrently consumed by the other (disjoint) subset of threads. A producer-consumer spatial partitioning pattern requires two one-sided synchronizations to manage a data buffer between the producer and consumer.

| Producer | Consumer |
| --- | --- |
| wait for buffer to be ready to be filled | signal buffer is ready to be filled |
| produce data and fill the buffer |  |
| signal buffer is filled | wait for buffer to be filled |
|  | consume data in filled buffer |

Producer threads wait for consumer threads to signal that the buffer is ready to be filled; however, consumer threads do not wait for this signal. Consumer threads wait for producer threads to signal that the buffer is filled; however, producer threads do not wait for this signal. For full producer/consumer concurrency this pattern has (at least) double buffering where each buffer requires two barriers.

**CUDA C++ cuda::barrier**

| ```cuda #include <cuda/barrier>  using barrier_t = cuda::barrier<cuda::thread_scope_block>;  __device__ void produce(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *in, int N) {   for (int i = 0; i < N / buffer_len; ++i)   {     ready[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */     /* produce, i.e., fill in, buffer_(i%2)  */     barrier_t::arrival_token token = filled[i % 2].arrive(); /* buffer_(i%2) is filled */   } }  __device__ void consume(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *out, int N) {   barrier_t::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */   barrier_t::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */   for (int i = 0; i < N / buffer_len; ++i)   {     filled[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */     /* consume buffer_(i%2) */     barrier_t::arrival_token token3 = ready[i % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */   } }  __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len) {   constexpr int warpSize = 32;    /* Shared memory buffer declared below is of size 2 * buffer_len      so that we can alternatively work between two buffers.      buffer_0 = buffer and buffer_1 = buffer + buffer_len */   __shared__ extern float buffer[];    /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,      while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */   #pragma nv_diag_suppress static_var_with_dynamic_init   __shared__ barrier_t bar[4];    if (threadIdx.x < 4)   {     init(bar + threadIdx.x, blockDim.x);   }   __syncthreads();    if (threadIdx.x < warpSize)   { produce(bar, bar + 2, buffer, buffer_len, in, N); }   else   { consume(bar, bar + 2, buffer, buffer_len, out, N); } } ``` |
| --- |

**CUDA C++ cuda::ptx**

| ```cuda #include <cuda/ptx>  __device__ void produce(barrier ready[], barrier filled[], float *buffer, int buffer_len, float *in, int N) {   for (int i = 0; i < N / buffer_len; ++i)   {     uint64_t token1 = cuda::ptx::mbarrier_arrive(ready[i % 2]);     while(!cuda::ptx::mbarrier_try_wait(&ready[i % 2], token1)) {} /* wait for buffer_(i%2) to be ready to be filled */     /* produce, i.e., fill in, buffer_(i%2)  */     uint64_t token2 = cuda::ptx::mbarrier_arrive(&filled[i % 2]); /* buffer_(i%2) is filled */   } }  __device__ void consume(barrier ready[], barrier filled[], float *buffer, buffer_len, float *out, int N) {   uint64_t token1 = cuda::ptx::mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */   uint64_t token2 = cuda::ptx::mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */   for (int i = 0; i < N / buffer_len; ++i)   {     uint64_t token3 = cuda::ptx::mbarrier_arrive(&filled[i % 2]);     while(!cuda::ptx::mbarrier_try_wait(&filled[i % 2], token3x)) {} /* wait for buffer_(i%2) to be filled */     /* consume buffer_(i%2) */     uint64_t token4 = cuda::ptx::mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */   } }  __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len) {   constexpr int warpSize = 32;    /* Shared memory buffer declared below is of size 2 * buffer_len      so that we can alternatively work between two buffers.      buffer_0 = buffer and buffer_1 = buffer + buffer_len */   __shared__ extern float buffer[];    /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,      while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */   #pragma nv_diag_suppress static_var_with_dynamic_init   __shared__ uint64_t bar[4];    if (threadIdx.x < 4)   {     cuda::ptx::mbarrier_init(bar + block.thread_rank(), block.size());   }   __syncthreads();    if (threadIdx.x < warpSize)   {  produce(bar, bar + 2, buffer, buffer_len, in, N); }   else   {  consume(bar, bar + 2, buffer, buffer_len, out, N); } } ``` |
| --- |

**CUDA C primitives**

| ```cuda #include <cuda_awbarrier_primitives.h>  __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N) {   for (int i = 0; i < N / buffer_len; ++i)   {     __mbarrier_token_t token1 = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */     while(!__mbarrier_try_wait(&ready[i % 2], token1, 1000)) {}     /* produce, i.e., fill in, buffer_(i%2)  */     __mbarrier_token_t token2 = __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */   } }  __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N) {   __mbarrier_token_t token1 = __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */   __mbarrier_token_t token2 = __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */   for (int i = 0; i < N / buffer_len; ++i)   {     __mbarrier_token_t token3 = __mbarrier_arrive(&filled[i % 2]);     while(!__mbarrier_try_wait(&filled[i % 2], token3, 1000)) {}     /* consume buffer_(i%2) */     __mbarrier_token_t token4 = __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */   } }  __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len) {   constexpr int warpSize = 32;    /* Shared memory buffer declared below is of size 2 * buffer_len      so that we can alternatively work between two buffers.      buffer_0 = buffer and buffer_1 = buffer + buffer_len */   __shared__ extern float buffer[];    /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,      while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */   #pragma nv_diag_suppress static_var_with_dynamic_init   __shared__ __mbarrier_t bar[4];    if (threadIdx.x < 4)   {     __mbarrier_init(bar + threadIdx.x, blockDim.x);   }   __syncthreads();    if (threadIdx.x < warpSize)   { produce(bar, bar + 2, buffer, buffer_len, in, N); }   else   { consume(bar, bar + 2, buffer, buffer_len, out, N); } } ``` |
| --- |

In this example, the first warp is specialized as the producer and the remaining warps are specialized as consumers. All producer and consumer threads participate (call `bar.arrive()` or `bar.arrive_and_wait()`) in each of the four barriers so the expected arrival counts are equal to `block.size()`.

A producer thread waits for the consumer threads to signal that the shared memory buffer can be filled. In order to wait for a barrier, a producer thread must first arrive on that `ready[i%2].arrive()` to get a token and then `ready[i%2].wait(token)` with that token. For simplicity, `ready[i%2].arrive_and_wait()` combines these operations.

```c++
bar.arrive_and_wait();
/* is equivalent to */
bar.wait(bar.arrive());
```

Producer threads compute and fill the ready buffer, they then signal that the buffer is filled by arriving on the filled barrier, `filled[i%2].arrive()`. A producer thread does not wait at this point, instead it waits until the next iteration’s buffer (double buffering) is ready to be filled.

A consumer thread begins by signaling that both buffers are ready to be filled. A consumer thread does not wait at this point, instead it waits for this iteration’s buffer to be filled, `filled[i%2].arrive_and_wait()`. After the consumer threads consume the buffer they signal that the buffer is ready to be filled again, `ready[i%2].arrive()`, and then wait for the next iteration’s buffer to be filled.
