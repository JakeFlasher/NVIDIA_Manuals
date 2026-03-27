---
title: "7.12.3. Some intuition on how to use token ordering"
section: "7.12.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#some-intuition-on-how-to-use-token-ordering"
---

### [7.12.3. Some intuition on how to use token ordering](https://docs.nvidia.com/cuda/tile-ir/latest/sections#some-intuition-on-how-to-use-token-ordering)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#some-intuition-on-how-to-use-token-ordering "Permalink to this headline")

When you have memory accesses to tiles with no overlap between them, it is generally safe to not order these with respect to each other in token order.
When you use a release operation, you need to token-order all memory events that must stay before the release to the release itself.
Similarly, when an acquire operation is used, all memory events which must remain after the acquire need to be token ordered after the acquire operation itself.

To reiterate: a user of **Tile IR** cannot rely on program dependencies of any form other than token dependencies to enforce ordering within a Tile Block Thread.
The **Tile IR** toolchain can remove dependencies through any possible complex reasoning, breaking dependencies which appear to be in the program syntax.
