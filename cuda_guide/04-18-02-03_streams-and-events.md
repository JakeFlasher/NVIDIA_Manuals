---
title: "4.18.2.3. Streams and Events"
section: "4.18.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#streams-and-events"
---

### [4.18.2.3. Streams and Events](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#streams-and-events)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#streams-and-events "Permalink to this headline")

CUDA _Streams_ and _Events_ allow control over dependencies between kernel launches: kernels launched into the same stream execute in-order, and events may be used to create dependencies between streams. Streams and events created on the device serve this exact same purpose.

Streams and events created within a grid exist within grid scope, but have undefined behavior when used outside of the grid where they were created. As described above, all work launched by a grid is implicitly synchronized when the grid exits; work launched into streams is included in this, with all dependencies resolved appropriately. The behavior of operations on a stream that has been modified outside of grid scope is undefined.

Streams and events created on the host have undefined behavior when used within any kernel, just as streams and events created by a parent grid have undefined behavior if used within a child grid.
