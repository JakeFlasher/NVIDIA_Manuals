---
title: "4.4.6.3. Invoke One"
section: "4.4.6.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#invoke-one"
---

### [4.4.6.3. Invoke One](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#invoke-one)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#invoke-one "Permalink to this headline")

Cooperative Groups provides an `invoke_one` function for use when a single thread must perform a serial portion of work on behalf of a group.
- `invoke_one` selects a single arbitrary thread from the calling group and uses that thread to call the supplied invocable function using the supplied arguments.
- `invoke_one_broadcast` is the same as `invoke_one` except the result of the call is also broadcast to all threads in the group.

The thread selection mechanism is not guaranteed to be deterministic.

The following example shows basic `invoke_one` utilization.

```c++
namespace cg = cooperative_groups;
cg::thread_block my_group = cg::this_thread_block();

// Ensure only one thread in the thread block prints the message
cg::invoke_one(my_group, []() {
   printf("Hello from one thread in the block!");
});

// Synchronize to make sure all threads wait until the message is printed
cg::sync(my_group);
```

Communication or synchronization within the calling group is not allowed inside the invocable function.
Communication with threads outside of the calling group is allowed.
