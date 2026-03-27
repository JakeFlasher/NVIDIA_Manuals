---
title: "4.8.5. Limitations and Known Issues"
section: "4.8.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#limitations-and-known-issues"
---

## [4.8.5. Limitations and Known Issues](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#limitations-and-known-issues)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#limitations-and-known-issues "Permalink to this headline")

1. The log buffer is limited to 100 entries. After this limit is reached, the oldest entries will be replaced and log dumps will contain a line noting the rollover.
2. Not all CUDA APIs are covered yet. This is an ongoing project to provide better usage error reporting for all APIs.
3. The Error Log Management log location (if given) will not be tested for validity until/unless a log is generated.
4. The Error Log Management APIs are currently only available via the CUDA Driver. Equivalent APIs will be added to the CUDA Runtime in a future release.
5. The log messages are not localized to any language and all provided logs are in US English.
