---
title: "2.5.4.6. Fatbin Compression"
section: "2.5.4.6"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#fatbin-compression"
---

### [2.5.4.6. Fatbin Compression](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#fatbin-compression)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#fatbin-compression "Permalink to this headline")

`nvcc` compresses the [fatbins](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-cubins-fatbins) stored in application or library binaries by default. Fatbin compression can be controlled using the following options:

- `-no-compress`: Disable the compression of the fatbin.
- `--compress-mode={default|size|speed|balance|none}`: Set the compression mode. `speed` focuses on fast decompression time, while `size` aims at reducing the fatbin size. `balance` provides a trade-off between speed and size. The default mode is `speed`. `none` disables compression.
