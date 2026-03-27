---
title: "4.11.2.2.5. Shared-Memory Bank Swizzling"
section: "4.11.2.2.5"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#shared-memory-bank-swizzling"
---

#### [4.11.2.2.5. Shared-Memory Bank Swizzling](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#shared-memory-bank-swizzling)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#shared-memory-bank-swizzling "Permalink to this headline")

By default, the TMA engine loads data to shared memory in the same order as it is laid out in global memory. However, this
layout may not be optimal for certain shared memory access patterns, as it could cause shared memory bank conflicts. To
improve performance and reduce bank conflicts, we can change the shared memory layout by applying a ‘swizzle pattern’.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a
bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is
used multiple times within a transaction, resulting in reduced bandwidth. See [Shared Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns).

To ensure that data is laid out in shared memory in such a way that user code can avoid shared memory bank conflicts,
the TMA engine can be instructed to ‘swizzle’ the data before storing it in shared memory and ‘unswizzle’ it when copying
the data back from shared memory to global memory. The tensor map encodes the ‘swizzle mode’ indicating which swizzle pattern is used.

**Example: Matrix Transpose**

An example is the transpose of a matrix where data is mapped from row to column first access. The data is stored row major in
global memory, but we want to also access it column wise in shared memory, which leads to bank conflicts. However, by using
the 128 bytes ‘swizzle’ mode and new shared memory indices, they are eliminated.

In the example, we load an 8x8 matrix of type `int4`, stored as row major in global memory to shared memory. Then, each set of eight
threads loads a row from the shared memory buffer and stores it to a column in a separate transpose shared memory buffer. This
results in an eight-way bank conflict when storing. Finally, the transpose buffer is written back to global memory.

To avoid bank conflicts, the `CU_TENSOR_MAP_SWIZZLE_128B` layout can be used. This layout matches the 128 bytes row length and
changes the shared memory layout in a way that both the column wise and row wise access don’t require the same banks per transaction.

The two tables,  [Figure 48](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-example1) and  [Figure 49](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-example2),  below show the normal and the swizzled shared memory layout of the 8x8 matrix of type `int4` and
its transpose matrix. The colors indicate which of the eight groups of four banks the matrix element is mapped to, and
the margin row and margin column list the global memory row and column indices. The entries show the shared memory
indices of the 16-byte matrix elements.

![The shared memory data layout without swizzle](images/______-______-____-_w________1.png)

Figure 48 In the shared memory data layout without swizzle, the shared memory indices are equivalent to the global memory indices.
Per load instruction, one row is read and stored in a column of the transpose buffer. Since all matrix elements of the
column in the transpose fall in the same bank, the store must be serialized, resulting in eight store transactions, giving
an eight-way bank conflict per stored column.[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id2 "Link to this image")

![The shared memory data layout with CU_TENSOR_MAP_SWIZZLE_128B swizzle.](images/______-______-____-_w________2.png)

Figure 49 The shared memory data layout with `CU_TENSOR_MAP_SWIZZLE_128B` swizzle. One row is stored in a column, each matrix
element is from a different bank for both the rows and columns, and so without any bank conflicts.[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id3 "Link to this image")

```c++
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
   }
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     int32_t tensor_coords[2] = { 0, 0 };
     ptx::cp_async_bulk_tensor(
       ptx::space_shared, ptx::space_global,
       &smem_buffer, &tensor_map, tensor_coords,
       cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* Matrix transpose
    *  When using the normal shared memory layout, there are eight
    *  8-way shared memory bank conflict when storing to the transpose.
    *  When enabling the 128-byte swizzle pattern and using the according access pattern,
    *  they are eliminated both for load and store. */
   for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // Wait for shared memory writes to be visible to TMA engine.
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();

   /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
    * it will 'unswizzle' the data. */
   if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer_tr);
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   }

   // Destroy barrier
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {4*8, 8};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```

**Remark.** This example is supposed to show the use of swizzle and ‘as-is’ is not performant nor does it scale beyond the given dimensions.

**Explanation.** During data transfer, the TMA engine shuffles the data according to the swizzle pattern, as described in the following
tables. These swizzle patterns define the mapping of the 16-byte chunks along the swizzle width to subgroups of four banks.
It is of type `CUtensorMapSwizzle` and has four options: none, 32 bytes, 64 bytes and 128 bytes. Note that the shared memory box’s
inner dimension must be less or equal to the span of the swizzle pattern.

**The Swizzle Modes**

As previously mentioned, there are four swizzle modes. The following tables show the different swizzle patterns, including the relation of the new
shared memory indices. The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

![An Overview of TMA Swizzle Patterns](images/______-______-____-_w________3.png)

Figure 50 An Overview of TMA Swizzle Patterns[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#id4 "Link to this image")

**Considerations.**  When applying a TMA swizzle pattern, it is crucial to adhere to specific memory requirements:

- **Global memory alignment:**
Global memory must be aligned to 128 bytes.
- **Shared memory alignment:**
For simplicity shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats. When the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself, there is an offset between the swizzle pattern and the shared memory.
See [comment](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#swizzle-pattern-pointer-offset-computation), below.
- **Inner dimension:**
The inner dimension of the shared memory block must meet the size requirements specified in [Table 25](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-swizzle-pattern-properties-and-requirements). If these
requirements are not met, the instruction is considered invalid. Additionally, if the swizzle width exceeds the inner dimension,
ensure that the shared memory is allocated to accommodate the full swizzle width.
- **Granularity:**
The granularity of swizzle mapping is fixed at 16 bytes. This means that data is organized and accessed in chunks
of 16 bytes, which must be considered when planning memory layout and access patterns.

**Swizzle Pattern Pointer Offset Computation**. Here, we describe how to determine the offset between the swizzle pattern and the shared memory, when the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself.
When using TMA, the shared memory is required to be aligned to 128 bytes. To find how many times the shared memory buffer relative to the swizzle pattern is shifted by that, apply the corresponding offset formula.

| Swizzle Mode | Offset Formula | Index Relation |
| --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%8` | `smem[y][x] <-> smem[y][((y+offset)%8)^x]` |
| CU_TENSOR_MAP_SWIZZLE_64B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%4` | `smem[y][x] <-> smem[y][((y+offset)%4)^x]` |
| CU_TENSOR_MAP_SWIZZLE_32B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%2` | `smem[y][x] <-> smem[y][((y+offset)%2)^x]` |

In [Figure 50](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#figure-swizzle-overview), this offset represents the initial row offset, thus, in the swizzle index calculation, it is added to the row index `y`.
The following snippet shows how to access the swizzled shared memory in the `CU_TENSOR_MAP_SWIZZLE_128B` mode.

```c++
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

**Summary.** The following [Table 25](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#table-swizzle-pattern-properties-and-requirements) summarizes the requirements and properties of the different swizzle patterns for Compute Capability 9.

| Pattern | Swizzle width | Shared box’s inner dimension | Repeats after | Shared memory alignment | Global memory alignment |
| --- | --- | --- | --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | 128 bytes | <=128 bytes | 1024 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_64B | 64 bytes | <=64 bytes | 512 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_32B | 32 bytes | <=32 bytes | 256 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_NONE (default) |  |  |  | 128 bytes | 16 bytes |
