---
title: "Epilogue config supported"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#epilogue-config-supported"
---

## [Epilogue config supported](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#epilogue-config-supported)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#epilogue-config-supported "Permalink to this headline")

**Table 14: Epilogue Dispatch Policy**

| Dense / Sparse | Legacy / Narrow Precision | 1/2 SM | Epilogue Dispatch Policy |
| --- | --- | --- | --- |
| Dense | Legacy & Narrow Precision | 1SM | `cutlass::epilogue::TmaWarpSpecialized1Sm` |
| Dense | Legacy & Narrow Precision | 1SM | `cutlass::epilogue::NoSmemWarpSpecialized1Sm` |
| Dense | Legacy & Narrow Precision | 2SM | `cutlass::epilogue::TmaWarpSpecialized2Sm` |
| Dense | Legacy & Narrow Precision | 2SM | `cutlass::epilogue::NoSmemWarpSpecialized2Sm` |
| Sparse | Legacy | 1SM | `cutlass::epilogue::TmaWarpSpecialized1Sm` |
| Sparse | Legacy | 1SM | `cutlass::epilogue::NoSmemWarpSpecialized1Sm` |
| Sparse | Legacy | 2SM | `cutlass::epilogue::TmaWarpSpecialized2Sm` |
| Sparse | Legacy | 2SM | `cutlass::epilogue::NoSmemWarpSpecialized2Sm` |
| Sparse | Narrow Precision (nvf4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmNvf4` |
| Sparse | Narrow Precision (nvf4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmNvf4` |
| Sparse | Narrow Precision (mxf4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmMxf4` |
| Sparse | Narrow Precision (mxf4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmMxf4` |
| Sparse | Narrow Precision (mxf8f6f4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4` |
| Sparse | Narrow Precision (mxf8f6f4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4` |

**Table 15: Epilogue PerSmTileShape_MNK**

| 1/2 SM | MMA tile Shape | PerSmTileShape_MNK |
| --- | --- | --- |
| 1SM | 64x64xMMA_TileShape_K | 64x64xMMA_TileShape_K |
| 1SM | 64x128xMMA_TileShape_K | 64x128xMMA_TileShape_K |
| 1SM | 64x192xMMA_TileShape_K | 64x192xMMA_TileShape_K |
| 1SM | 64x256xMMA_TileShape_K | 64x256xMMA_TileShape_K |
| 1SM | 128x64xMMA_TileShape_K | 128x64xMMA_TileShape_K |
| 1SM | 128x128xMMA_TileShape_K | 128x128xMMA_TileShape_K |
| 1SM | 128x192xMMA_TileShape_K | 128x192xMMA_TileShape_K |
| 1SM | 128x256xMMA_TileShape_K | 128x256xMMA_TileShape_K |
| 2SM | 128x64xMMA_TileShape_K | 64x64xMMA_TileShape_K |
| 2SM | 128x128xMMA_TileShape_K | 64x128xMMA_TileShape_K |
| 2SM | 128x192xMMA_TileShape_K | 64x192xMMA_TileShape_K |
| 2SM | 128x256xMMA_TileShape_K | 64x256xMMA_TileShape_K |
| 2SM | 256x64xMMA_TileShape_K | 128x64xMMA_TileShape_K |
| 2SM | 256x128xMMA_TileShape_K | 128x128xMMA_TileShape_K |
| 2SM | 256x192xMMA_TileShape_K | 128x192xMMA_TileShape_K |
| 2SM | 256x256xMMA_TileShape_K | 128x256xMMA_TileShape_K |

MMA_TileShape_K is is generally 4 * MMA-Instruction-K. It depends on the config we defined in MMA tile shapes supported section.
