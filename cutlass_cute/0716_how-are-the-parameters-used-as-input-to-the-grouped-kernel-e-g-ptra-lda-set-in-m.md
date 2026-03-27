---
title: "How are the parameters used as input to the grouped kernel (e.g., ptrA, lda) set in my application?"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html#how-are-the-parameters-used-as-input-to-the-grouped-kernel-e-g-ptra-lda-set-in-my-application"
---

### [How are the parameters used as input to the grouped kernel (e.g., ptrA, lda) set in my application?](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#how-are-the-parameters-used-as-input-to-the-grouped-kernel-e-g-ptra-lda-set-in-my-application)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#how-are-the-parameters-used-as-input-to-the-grouped-kernel-e-g-ptra-lda-set-in-my-application "Permalink to this headline")

If these are set by a previous kernel running on
the device (rather than by the host), you likely want to use `kDeviceOnly`,
as this will minimize additional host-device communication.
