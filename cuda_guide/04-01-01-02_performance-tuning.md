---
title: "4.1.1.2. Performance Tuning"
section: "4.1.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#performance-tuning"
---

### [4.1.1.2. Performance Tuning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#performance-tuning)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#performance-tuning "Permalink to this headline")

In order to achieve good performance with unified memory, it is important to:

- understand how paging works on your system, and how to avoid unnecessary page faults
- understand the various mechanisms allowing you to keep data local to the accessing processor
- consider tuning your application for the granularity of memory transfers of your system.

As general advice, performance hints (see [Performance Hints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#um-perf-hints))
might provide improved performance, but using them incorrectly might degrade performance
compared to the default behavior.
Also note that any hint has a performance cost associated with it on the host,
thus useful hints must at the very least improve performance enough to overcome this cost.
