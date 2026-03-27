---
title: "3.5.5.3. Error Log Management"
section: "3.5.5.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#error-log-management"
---

### [3.5.5.3. Error Log Management](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#error-log-management)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#error-log-management "Permalink to this headline")

[Error log management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management) provides utilities for handling and logging errors from CUDA APIs. Setting a single environment variable `CUDA_LOG_FILE` enables capturing CUDA errors directly to stderr, stdout, or a file.  Error log management also enables applications to register a callback which is triggered when CUDA encounters an error. [Section 4.8](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management) provides more details on error log management.
