---
title: "9.1.5. Mechanism"
section: "9.1.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#mechanism"
---

### [9.1.5. Mechanism](https://docs.nvidia.com/cuda/tile-ir/latest/sections#mechanism)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#mechanism "Permalink to this headline")

To synthesize scope metadata, the SynthesizeDebugInfoScopes pass can be added to any **Tile IR** pass pipeline or invoked via Python as follows:

```python
pm = passmanager.PassManager(context=module.context)
full_pass_string = f"cuda_tile.module({"synthesize-debug-info-scopes"})"
pm = pm.parse(full_pass_string, context=module.context)
pm.run(module.operation.regions[0].blocks[0].operations[0])
```
