---
title: "5.1.2.3. Feature Set Compiler Targets"
section: "5.1.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#feature-set-compiler-targets"
---

### [5.1.2.3. Feature Set Compiler Targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#feature-set-compiler-targets)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#feature-set-compiler-targets "Permalink to this headline")

There are three sets of compute features which the compiler can target:

**Baseline Feature Set**: The predominant set of compute features that are introduced with the intent to be available for subsequent compute architectures.  These features and their availability are summarized in [Table 29](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability).

**Architecture-Specific Feature Set**: A small and highly specialized set of features called architecture-specific, that are introduced to accelerate specialized operations, which are not guaranteed to be available or might change significantly on subsequent compute architectures.  These features are summarized in the respective “Compute Capability #.#” subsections.  The architecture-specific feature set is a superset of the family-specific feature set.  Architecture-specific compiler targets were introduced with Compute Capability 9.0 devices and are selected by using an **a** suffix in the compilation target, for example by specifying `compute_100a` or `compute_120a` as the compute target.

**Family-Specific Feature Set**: Some architecture-specific features are common to GPUs of more than one compute capability. These features are summarized in the respective “Compute Capability #.#” subsections. With a few exceptions, later-generation devices with the same major compute capability are in the same family. [Table 28](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#compute-capabilities-family-specific-compatibility) indicates the compatibility of family-specific targets with device compute capability, including exceptions. The family-specific feature set is a superset of the baseline feature set.  Family-specific compiler targets were introduced with Compute Capability 10.0 devices and are selected by using an **f** suffix in the compilation target, for example by specifying `compute_100f` or `compute_120f` as the compute target.

All devices starting from compute capability 9.0 have a set of features that are architecture-specific. To utilize the complete set of these features on a specific GPU, the architecture-specific compiler target with the suffix **a** must be used. Additionally, starting from compute capability 10.0, there are sets of features that appear in multiple devices with different minor compute capabilities. These sets of instructions are called family-specific features, and the devices which share these features are said to be part of the same family. The family-specific features are a subset of the architecture-specific features that are shared by all members of that GPU family. The family-specific compiler target with the suffix **f** allows the compiler to generate code that uses this common subset of architecture-specific features.

For example:

- The `compute_100` compilation target does not allow the use of architecture-specific features.  This target will be compatible with all devices of compute capability 10.0 and later.
- The `compute_100f` _family-specific_ compilation target allows the use of the subset of architecture-specific features that are common across the GPU family. This target will only be compatible with devices that are part of the GPU family. In this example, it is compatible with devices of Compute Capability 10.0 and Compute Capability 10.3. The features available in the family-specific `compute_100f` target are a superset of the features available in the baseline `compute_100` target.
- The `compute_100a` _architecture-specific_ compilation target allows the use of the complete set of architecture-specific features in Compute Capability 10.0 devices. This target will only be compatible with devices of Compute Capability 10.0 and no others. The features available in the `compute_100a` target form a superset of the features available in the `compute_100f` target.

| Compilation Target | Compatible with Compute Capability |  |
| --- | --- | --- |
| `compute_100f` | 10.0 | 10.3 |
| `compute_103f` | 10.3 [^[1]] |  |
| `compute_110f` | 11.0 [^[1]] |  |
| `compute_120f` | 12.0 | 12.1 |
| `compute_121f` | 12.1 [^[1]] |  |

[^[1]]: ([1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#id2),[2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#id3),[3](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#id4))
Some families only contain a single member when they are created. They may be expanded in the future to include more devices.
