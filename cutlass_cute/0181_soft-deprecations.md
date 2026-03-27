---
title: "Soft Deprecations"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/deprecation.html#soft-deprecations"
---

## [Soft Deprecations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#soft-deprecations)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#soft-deprecations "Permalink to this headline")

**Version 4.2.1**

- `cute.arch.warpgroup_reg_alloc` and `cute.arch.warpgroup_reg_dealloc`
→ Scheduled for deprecation. Use `cute.arch.setmaxregister_increase` and `cute.arch.setmaxregister_decrease` instead.
- `alignment` argument in `CooperativeGroup` constructor
→ Scheduled for deprecation. It was unused; no replacement is suggested.
