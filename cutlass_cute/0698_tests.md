---
title: "Tests"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#tests"
---

## [Tests](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tests)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tests "Permalink to this headline")

Test programs for CUTLASS. Tests are organized hierarchically, mirroring the organization of source files.

```console
test/                        # unit tests for CUTLASS Template Library
  unit/
    arch/
    core/
    gemm/
      device/
      kernel/
      thread/
      threadblock/
      warp/
    reduction/
      kernel/
      thread/
    transform/
      threadblock/
      *
```

Tests can be built and run at the top level scope by invoking `make test_unit` or by building
and explicitly executing each individual target, e.g. `cutlass_test_unit_gemm_device`.

Tests are configured to specify appropriate GTest filter strings to avoid running except on
architectures where they are expected to pass. Thus, no tests should fail. The actual number
of tests run may vary over time as more are added.
