---
title: "Operating Systems"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/overview.html#operating-systems"
---

## [Operating Systems](https://docs.nvidia.com/cutlass/latest#operating-systems)[](https://docs.nvidia.com/cutlass/latest/#operating-systems "Permalink to this headline")

We have tested the following environments.

| **Operating System** | **Compiler** |
| --- | --- |
| Ubuntu 18.04 | GCC 7.5.0 |
| Ubuntu 20.04 | GCC 10.3.0 |
| Ubuntu 22.04 | GCC 11.2.0 |

Note: GCC 8.5.0 has known regressions regarding fold expressions and overloaded operators. Using GCC 7.5.0 or (preferred) GCC >= 9 is recommended.

Note: CUTLASS 3.x builds are known to be down on Windows platforms for all CUDA toolkits.
CUTLASS team is working on a fix.
