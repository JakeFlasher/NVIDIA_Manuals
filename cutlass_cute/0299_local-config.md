---
title: "Local Config"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html#local-config"
---

### [Local Config](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#local-config)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#local-config "Permalink to this headline")

Local config is needed to specify per-project settings, especially include paths. An example is:

```console
CompileFlags:
  Add:
    - -I</absolute/path/to/cutlass>/include/
    - -I</absolute/path/to/cutlass>/tools/util/include/
    - -I</absolute/path/to/cutlass>/cutlass/examples/common/
```

Note that absolute paths are needed since clangd doesn’t support relative paths.
