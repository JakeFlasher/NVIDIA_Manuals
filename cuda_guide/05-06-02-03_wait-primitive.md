---
title: "5.6.2.3. Wait Primitive"
section: "5.6.2.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#wait-primitive"
---

### [5.6.2.3. Wait Primitive](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#wait-primitive)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#wait-primitive "Permalink to this headline")

```cuda
void __pipeline_wait_prior(size_t N);
```

- Let `{0, 1, 2, ..., L}` be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread.
- Wait for completion of batches _at least_ up to and including `L-N`.
