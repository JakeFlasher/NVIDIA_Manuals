---
title: "4.8.2. Activation"
section: "4.8.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#activation"
---

## [4.8.2. Activation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#activation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#activation "Permalink to this headline")

Set the _CUDA_LOG_FILE_ environment variable. Acceptable values are _stdout_, _stderr_, or a valid path on the system to write a file.
The log buffer can be dumped via API even if _CUDA_LOG_FILE_ was not set before program execution.
NOTE: An error-free execution may not print any logs.
