---
title: "5.1.2.1. Architecture-Specific Features"
section: "5.1.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#architecture-specific-features"
---

### [5.1.2.1. Architecture-Specific Features](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#architecture-specific-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#architecture-specific-features "Permalink to this headline")

Beginning with devices of Compute Capability 9.0, specialized compute features that are introduced with an architecture may not be guaranteed to be available on all subsequent compute capabilities. These features are called _architecture-specific_ features and target acceleration of specialized operations, such as Tensor Core operations, which are not intended for all classes of compute capabilities or may significantly change in future generations.  Code must be compiled with an architecture-specific compiler target (see [Feature Set Compiler Targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compute-capabilities-feature-set-compiler-targets)) to enable architecture-specific features. Code compiled with an architecture-specific compiler target can only be run on the exact compute capability it was compiled for.
