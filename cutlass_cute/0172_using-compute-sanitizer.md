---
title: "Using Compute-Sanitizer"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#using-compute-sanitizer"
---

### [Using Compute-Sanitizer](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#using-compute-sanitizer)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#using-compute-sanitizer "Permalink to this headline")

For detecting memory errors and race conditions:

```bash
compute-sanitizer --some_options python your_dsl_code.py
```

Please refer to the [compute-sanitizer documentation](https://developer.nvidia.com/compute-sanitizer) for more details.
