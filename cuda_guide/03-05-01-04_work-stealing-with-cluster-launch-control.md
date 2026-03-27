---
title: "3.5.1.4. Work Stealing with Cluster Launch Control"
section: "3.5.1.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html#work-stealing-with-cluster-launch-control"
---

### [3.5.1.4. Work Stealing with Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#work-stealing-with-cluster-launch-control)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#work-stealing-with-cluster-launch-control "Permalink to this headline")

Work stealing is a technique for maintaining utilization in uneven workloads where workers that have completed their work can ‘steal’ tasks from other workers. Cluster launch control, a feature introduced in compute capability 10.0 (Blackwell), gives kernels direct control over in-flight block scheduling so they can adapt to uneven workloads in real time. A thread block can cancel the launch of another thread block or cluster that has not yet started, claim its index, and immediately begin executing the stolen work. This work-stealing flow keeps SMs busy and cuts idle time under irregular data or runtime variation—delivering finer-grained load balancing without relying on the hardware scheduler alone.

[Section 4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#cluster-launch-control) provides details on how to use this feature.
