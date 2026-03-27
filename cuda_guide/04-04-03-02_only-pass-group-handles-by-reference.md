---
title: "4.4.3.2. Only Pass Group Handles by Reference"
section: "4.4.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#only-pass-group-handles-by-reference"
---

### [4.4.3.2. Only Pass Group Handles by Reference](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#only-pass-group-handles-by-reference)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#only-pass-group-handles-by-reference "Permalink to this headline")

It is recommended that you pass group handles by reference to functions when passing a group handle into a function.
Group handles must be initialized at declaration time, as there is no default constructor.
Copy-constructing group handles is discouraged.
