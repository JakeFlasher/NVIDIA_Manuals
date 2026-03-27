---
title: "Loading in Python"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#loading-in-python"
---

### [Loading in Python](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#loading-in-python)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#loading-in-python "Permalink to this headline")

Load pre-compiled object files or shared libraries into Python for execution.

```python
import cutlass.cute as cute
import torch
from cutlass.cute import from_dlpack
import cutlass.cute.cuda as cuda

# Load module from object file
module = cute.runtime.load_module("./artifacts/print_tensor_example.o")
# or
module = cute.runtime.load_module("./artifacts/libprint_tensor_example.so")

# Prepare data
a = torch.arange(160, dtype=torch.float32, device="cuda").reshape(16, 10)
a_cute = from_dlpack(a).mark_layout_dynamic()
stream = cuda.CUstream(0)

# Call the function (no JIT compilation needed!)
module.print_tensor(a_cute, stream=stream)

# This will fail because 'non_existing_api' was not exported:
# module.non_existing_api()
```
