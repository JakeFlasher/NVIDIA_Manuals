---
title: "Interpreting synclog output"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#interpreting-synclog-output"
---

### [Interpreting synclog output](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#interpreting-synclog-output)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#interpreting-synclog-output "Permalink to this headline")

The synclog.txt file will contain runtime information about synchronization events. Here’s a sample output snippet:

```console
synclog start
synclog at 1: cluster_barrier_init line=281 time=1725400116233388736 thread=0,0,0 block=0,0,0 smem_addr=197632 arrive_count=1
synclog at 13: fence_barrier_init line=583 time=1725400116233388768 thread=32,0,0 block=0,0,0
...
```

Each line in the main body follows this format:

```console
synclog at [synclog_at]: [header] line=[line] thread=[threadIdx.xyz] block=[blockIdx.xyz]
```

- `synclog at`: Address in the `synclog` output buffer (in bytes). Output exceeding 2^26 bytes is discarded.
- `header`: Name of the synchronization event.
- `line`: Code line number of the synchronization operation calling into `synclog`.

Additional information may appear at the end of each line, such as shared memory address, phase bit, and arrive count. For more detailed information on `synclog` output, refer to [synclog.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/synclog.hpp) in the CUTLASS source code.

Please note that `synclog` is an experimental feature, and its functionality is not always guaranteed. We encourage its use in custom kernels and CUTLASS examples, though it is known to be incompatible with profiler kernels.
