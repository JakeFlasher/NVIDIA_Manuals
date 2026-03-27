---
title: "Known Issues"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/known_issues.html#known-issues"
---

# [Known Issues](https://docs.nvidia.com/cuda/cutile-python#known-issues)[](https://docs.nvidia.com/cuda/cutile-python/#known-issues "Permalink to this headline")

1. FP8 Torch Tensor requires *torch>=2.10*. Older version of PyTorch does not support converting fp8
datatype through *dlpack* protocol and will [leak memory](https://github.com/pytorch/pytorch/issues/171820)
when conversion to dlpack tensor fails.
