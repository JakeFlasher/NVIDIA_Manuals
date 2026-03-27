---
title: "3.1.3. More on Streams and Events"
section: "3.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#more-on-streams-and-events"
---

## [3.1.3. More on Streams and Events](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#more-on-streams-and-events)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#more-on-streams-and-events "Permalink to this headline")

[CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams) introduced the basics of CUDA streams. By default, operations submitted on a given CUDA stream are serialized: one cannot start executing until the previous one has completed. The only exception is the recently added [Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) feature. Having multiple CUDA streams is a way to enable concurrent execution; another way is using [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs). The two approaches can also be combined.

Work submitted on different CUDA streams *may* execute concurrently under specific circumstances, e.g., if there are no event dependencies, if there is no implicit synchronization, if there are sufficient resources, etc.

Independent operations from different CUDA streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in between them, unless the streams are non-blocking CUDA streams. These are streams created with `cudaStreamCreateWithFlags()` runtime API with the `cudaStreamNonBlocking` flag. To improve potential for concurrent GPU work execution, it is recommended that the user creates non-blocking CUDA streams.

It is also recommended that the user selects the least general synchronization option that is sufficient for their problem.
For example, if the requirement is for the CPU to wait (block) for all work on a specific CUDA stream to complete,
using `cudaStreamSynchronize()` for that stream would be preferable to `cudaDeviceSynchronize()`, as the latter would unnecessarily wait for GPU work on all CUDA streams of the device to complete.
And if the requirement is for the CPU to wait without blocking, then using `cudaStreamQuery()` and checking its return value, in a polling loop, may be preferable.

A similar synchronization effect can also be achieved with CUDA events ([CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)), e.g., by recording an event on that stream and calling
`cudaEventSynchronize()` to wait, in a blocking manner, for the work captured in that event to complete.
Again, this would be preferable and more focused than using `cudaDeviceSynchronize()`.
Calling `cudaEventQuery()` and checking its return value, e.g., in a polling loop, would be a non-blocking alternative.

The choice of the explicit synchronization method is particularly important if this operation happens in the application’s critical path.
[Table 4](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#table-streams-event-sync-summary) provides a high-level summary of various synchronization options
with the host.

|  | Wait for specific stream | Wait for specific event | Wait for everything on the device |
| --- | --- | --- | --- |
| Non-blocking (would need a polling loop) | cudaStreamQuery() | cudaEventQuery() | N/A |
| Blocking | cudaStreamSynchronize() | cudaEventSynchronize() | cudaDeviceSynchronize() |

For synchronization, i.e., to express dependencies, between CUDA streams, use of non-timing CUDA events is recommended, as described in [CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events).
A user can call `cudaStreamWaitEvent()` to force future submitted operations on a specific stream to wait for the completion of a previously recorded event (e.g., on another stream).
Note that for any CUDA API waiting or querying an event, it is the responsibility of the user to ensure the cudaEventRecord API has been already called,
as a non-recorded event will always return success.

CUDA events carry, by default, timing information, as they can be used in `cudaEventElapsedTime()` API calls.
However, a CUDA event that is solely used to express dependencies across streams does not need timing information.
For such cases, it is recommended to create events with timing information disabled for improved performance. This is possible using `cudaEventCreateWithFlags()` API
with the `cudaEventDisableTiming` flag.
