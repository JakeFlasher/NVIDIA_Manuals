---
title: "4.2.6.1.3. Device Graph Update"
section: "4.2.6.1.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-update"
---

#### [4.2.6.1.3. Device Graph Update](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#device-graph-update)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#device-graph-update "Permalink to this headline")

Device graphs can only be updated from the host, and must be re-uploaded to the device upon executable graph update in order for the changes to take effect. This can be achieved using the same methods outlined in Section [device-graph-upload](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cuda-graphs-device-graph-upload). Unlike host graphs, launching a device graph from the device while an update is being applied will result in undefined behavior.
