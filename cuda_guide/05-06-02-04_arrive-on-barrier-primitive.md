---
title: "5.6.2.4. Arrive On Barrier Primitive"
section: "5.6.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#arrive-on-barrier-primitive"
---

### [5.6.2.4. Arrive On Barrier Primitive](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#arrive-on-barrier-primitive)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#arrive-on-barrier-primitive "Permalink to this headline")

```cuda
void __pipeline_arrive_on(__mbarrier_t* bar);
```

- `bar` points to a barrier in shared memory.
- Increments the barrier arrival count by one, when all memcpy_async operations sequenced before this call have completed, the arrival count is decremented by one and hence the net effect on the arrival count is zero. It is user’s responsibility to make sure that the increment on the arrival count does not exceed `__mbarrier_maximum_count()`.
