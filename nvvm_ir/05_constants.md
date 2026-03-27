---
title: "5. Constants"
section: "5"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#constants"
---

# [5. Constants](https://docs.nvidia.com/cuda/nvvm-ir-spec#constants)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#constants "Permalink to this headline")

Supported, except for the following:

- `Token constants` are not supported.
- `blockaddress(@function, %block)` is not supported.
- For a constant expression that is used as the initializer of a global variable `@g1`, if the constant expression contains a global identifier `@g2`, then the constant expression is supported if it can be reduced to the form of `bitcast+offset`, where offset is an integer number (including `0`)
