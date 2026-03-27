---
title: "4.11.2.2.4. Creating a Template Tensor Map Value Using the Driver API"
section: "4.11.2.2.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#creating-a-template-tensor-map-value-using-the-driver-api"
---

#### [4.11.2.2.4. Creating a Template Tensor Map Value Using the Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#creating-a-template-tensor-map-value-using-the-driver-api)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#creating-a-template-tensor-map-value-using-the-driver-api "Permalink to this headline")

The following code creates a minimal tiled-type tensor map that can be subsequently modified on device.

```cuda
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
```
