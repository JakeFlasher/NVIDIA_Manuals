---
title: "4.2.5.6. Peer Access"
section: "4.2.5.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#peer-access"
---

### [4.2.5.6. Peer Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#peer-access)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#peer-access "Permalink to this headline")

Graph allocations can be configured for access from multiple GPUs, in which case CUDA will map the allocations onto the peer GPUs as required. CUDA allows graph allocations requiring different mappings to reuse the same virtual address. When this occurs, the address range is mapped onto all GPUs required by the different allocations. This means an allocation may sometimes allow more peer access than was requested during its creation; however, relying on these extra mappings is still an error.
