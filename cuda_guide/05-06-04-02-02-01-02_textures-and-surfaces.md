---
title: "5.6.4.2.2.1.2. Textures and Surfaces"
section: "5.6.4.2.2.1.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#textures-and-surfaces"
---

###### [5.6.4.2.2.1.2. Textures and Surfaces](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#textures-and-surfaces)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#textures-and-surfaces "Permalink to this headline")

> The device runtime does not allow creation or destruction of texture or surface objects from within device code. Texture and surface objects created from the host may be used and passed around freely on the device. Regardless of where they are created, dynamically created texture objects are always valid and may be passed to child kernels from a parent.

> **Note**
>
> The device runtime does not support legacy module-scope (i.e., compute capability 2.0 or Fermi-style) textures and surfaces within a kernel launched from the device. Module-scope (legacy) textures may be created from the host and used in device code as for any kernel, but may only be used by a top-level kernel (i.e., the one which is launched from the host).
