---
title: "5.6.2.2. Commit Primitive"
section: "5.6.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#commit-primitive"
---

### [5.6.2.2. Commit Primitive](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#commit-primitive)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#commit-primitive "Permalink to this headline")

```cuda
void __pipeline_commit();
```

- Commit submitted `memcpy_async` to the pipeline as the current batch.
