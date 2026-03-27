---
title: "5.6.3.6.2. sync"
section: "5.6.3.6.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-callable-apis--sync"
---

#### [5.6.3.6.2. sync](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#sync)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#sync "Permalink to this headline")

```c++
static void T::sync();

template <typename T>
void sync(T& group);
```

`sync` synchronizes the threads named in the group. Group type `T` can be any of the existing group types, as all of them support synchronization. Its available as a member function in every group type or as a free function taking a group as parameter.
