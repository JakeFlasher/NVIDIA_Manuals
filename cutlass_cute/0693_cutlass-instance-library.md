---
title: "CUTLASS Instance Library"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#cutlass-instance-library"
---

## [CUTLASS Instance Library](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-instance-library)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-instance-library "Permalink to this headline")

The CUTLASS Instance Library contains instantiations of the above CUTLASS templates covering supported configurations,
data types, block structure, and tile sizes. These instantiations are procedurally generated using a set of
scripts to span the design space.

```console
tools/
  library/                   # static/dynamic library containing all kernel instantiations of interest
                             # (with some build-level filter switches to compile specific subsets)

    include/
      cutlass/
        library/             # header files for CUTLASS Deliverables Library (in cutlass::library:: namespace)

          handle.h           # implements a host-side API for launching kernels, similar to cuBLAS
          library.h          # defines enums and structs to describe the tiled structure of operator instances
          manifest.h         # collection of all instances

    src/

python/
    cutlass_library/       # scripts to procedurally generate CUTLASS template instances

      gemm_operations.py
      library.py
      generator.py            # entry point of procedural generation scripts - invoked by cmake
      manifest.py
```

When CMake is executed, the CUTLASS Instance Library generator scripts are executed to construct a set of
instantiations in `build/tools/library/generated/`.
