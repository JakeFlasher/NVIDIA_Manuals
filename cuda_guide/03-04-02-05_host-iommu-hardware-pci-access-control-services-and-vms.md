---
title: "3.4.2.5. Host IOMMU Hardware, PCI Access Control Services, and VMs"
section: "3.4.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms"
---

### [3.4.2.5. Host IOMMU Hardware, PCI Access Control Services, and VMs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced#host-iommu-hardware-pci-access-control-services-and-vms)[](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/#host-iommu-hardware-pci-access-control-services-and-vms "Permalink to this headline")

On Linux specifically, CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer.
However, CUDA and the display driver do support IOMMU via virtual machine pass through.
The IOMMU must be disabled when running Linux on a bare metal system to prevent silent device memory corruption.
Conversely, the IOMMU should be enabled and the VFIO driver be used for PCIe pass through for virtual machines.

On Windows the IOMMU limitation above does not exist.

See also [Allocating DMA Buffers on 64-bit Platforms](https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html).

Additionally, PCI Access Control Services (ACS) can be enabled on systems that support IOMMU.
The PCI ACS feature redirects all PCI point-to-point traffic through the CPU root complex, which can cause significant performance loss due to the reduction in overall bisection bandwidth.
