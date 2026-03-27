---
title: "6. Debug Information"
section: "6"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#debug-information"
---

# [6. Debug Information](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#debug-information)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#debug-information "Permalink to this headline")

Debug information is encoded in DWARF (Debug With Arbitrary Record Format).

## [6.1. Generation of Debug Information](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#generation-of-debug-information)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#generation-of-debug-information "Permalink to this headline")

The responsibility for generating debug information is split between the PTX producer and the PTX-to-SASS backend. The PTX producer is responsible for emitting binary DWARF into the PTX file, using the .section and .b8-.b16-.b32-and-.b64 directives in PTX. This should contain the .debug_info and .debug_abbrev sections, and possibly optional sections .debug_pubnames and .debug_aranges. These sections are standard DWARF2 sections that refer to labels and registers in the PTX.

The PTX-to-SASS backend is responsible for generating the .debug_line section from the .file and .loc directives in the PTX file. This section maps source lines to SASS addresses. The backend also generates the .debug_frame section.

## [6.2. CUDA-Specific DWARF Definitions](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#cuda-specific-dwarf-definitions)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#cuda-specific-dwarf-definitions "Permalink to this headline")

In order to support debugging of multiple memory segments, address class codes are defined to reflect the memory space of variables. The address-class values are emitted as the DW_AT_address_class attribute for all variable and parameter Debugging Information Entries. The address class codes are defined in the below table.

| Code | Value | Description |
| --- | --- | --- |
| ADDR_code_space | 1 | Code storage |
| ADDR_reg_space | 2 | Register storage |
| ADDR_sreg_space | 3 | Special register storage |
| ADDR_const_space | 4 | Constant storage |
| ADDR_global_space | 5 | Global storage |
| ADDR_local_space | 6 | Local storage |
| ADDR_param_space | 7 | Parameter storage |
| ADDR_shared_space | 8 | Shared storage |
| ADDR_surf_space | 9 | Surface storage |
| ADDR_tex_space | 10 | Texture storage |
| ADDR_tex_sampler_space | 11 | Texture sampler storage |
| ADDR_generic_space | 12 | Generic-address storage |
