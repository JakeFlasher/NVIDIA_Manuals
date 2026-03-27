---
title: "Architecture-specific configuration"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/performance.html#architecture-specific-configuration"
---

## [Architecture-specific configuration](https://docs.nvidia.com/cuda/cutile-python#architecture-specific-configuration)[](https://docs.nvidia.com/cuda/cutile-python/#architecture-specific-configuration "Permalink to this headline")

```
_`class`_`cuda.tile.``ByTarget`(_`*`_, _`default``=``UNSPECIFIED`_, _`**``value_by_target`_)[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.ByTarget "Link to this definition")
```

Type used to specify a value that depends on the target GPU architecture.

**Parameters:**
: - **default** – The fallback value to use when the target GPU architecture is not explicitly
listed in `value_by_target`.
- **value_by_target** – Mapping from GPU architecture name to value. Keys must be strings of
the form `"sm_<major><minor>"`, such as `"sm_100"` or `"sm_120"`.

**Examples**

Use one `num_ctas` value for all architectures:

```python
from cuda.tile import kernel, ByTarget

@kernel(num_ctas=8)
def kernel_fn(x):
    ...
```

Use different `num_ctas` values for specific architectures, and a
fallback value for all others:

```python
from cuda.tile import kernel, ByTarget

@kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def kernel_fn(x):
    ...
```

See [Tile Kernels](https://docs.nvidia.com/cuda/cutile-python/execution.html#tile-kernels) for the full description of kernel configuration
parameters such as `num_ctas`, `occupancy` and `opt_level`. Any of
these options may be given as a [`ByTarget`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.ByTarget "cuda.tile.ByTarget") value to specialize them
for different GPU architectures.
