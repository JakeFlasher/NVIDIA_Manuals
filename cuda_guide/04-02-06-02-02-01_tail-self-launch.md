---
title: "4.2.6.2.2.1. Tail Self-launch"
section: "4.2.6.2.2.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#tail-self-launch"
---

##### [4.2.6.2.2.1. Tail Self-launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#tail-self-launch)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#tail-self-launch "Permalink to this headline")

It is possible for a device graph to enqueue itself for a tail launch, although a given graph can only have one self-launch enqueued at a time. In order to query the currently running device graph so that it can be relaunched, a new device-side function is added:

```cuda
cudaGraphExec_t cudaGetCurrentGraphExec();
```

This function returns the handle of the currently running graph if it is a device graph. If the currently executing kernel is not a node within a device graph, this function will return NULL.

Below is sample code showing usage of this function for a relaunch loop:

```cuda
__device__ int relaunchCount = 0;

__global__ void relaunchSelf() {
    int relaunchMax = 100;

    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }

        relaunchCount++;
    }
}
```
