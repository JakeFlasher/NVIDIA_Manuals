---
title: "7. Example"
section: "7"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#example"
---

# [7. Example](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#example)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#example "Permalink to this headline")

The following is example PTX with debug information for implementing the following program that makes a call:

```c++
__device__ __noinline__ int foo (int i, int j)
{
  return i+j;
}

__global__ void test (int *p)
{
  *p = foo(1, 2);
}
```

The resulting PTX would be something like:

```c++
.version 4.2
.target sm_20, debug
.address_size 64

 .file   1 "call_example.cu"

.visible .func  (.param .b32 func_retval0) // return value
_Z3fooii(
        .param .b32 _Z3fooii_param_0, // parameter "i"
        .param .b32 _Z3fooii_param_1) // parameter "j"
{
        .reg .s32       %r<4>;
        .loc 1 1 1      // following instructions are for line 1

func_begin0:
        ld.param.u32    %r1, [_Z3fooii_param_0]; // load 1st param
        ld.param.u32    %r2, [_Z3fooii_param_1]; // load 2nd param
        .loc    1 3 1   // following instructions are for line 3
        add.s32         %r3, %r1, %r2;
        st.param.b32    [func_retval0+0], %r3; // store return value
        ret;
func_end0:
}

.visible .entry _Z4testPi(
        .param .u64 _Z4testPi_param_0) // parameter *p
{
        .reg .s32       %r<4>;
        .reg .s64       %rd<2>;
        .loc 1 6 1

func_begin1:
        ld.param.u64    %rd1, [_Z4testPi_param_0]; // load *p
        mov.u32         %r1, 1;
        mov.u32         %r2, 2;
        .loc    1 8 9
        .param .b32 param0;
        st.param.b32    [param0+0], %r1; // store 1
        .param .b32 param1;
        st.param.b32    [param1+0], %r2; // store 2
        .param .b32 retval0;
        call.uni (retval0), _Z3fooii, ( param0, param1); // call foo
        ld.param.b32    %r3, [retval0+0]; // get return value
        st.u32  [%rd1], %r3;              // *p = return value
        .loc    1 9 2
        ret;
func_end1:
}
```

```c++
.section .debug_info {
 .b32 262
 .b8 2, 0
 .b32 .debug_abbrev
 .b8 8, 1, 108, 103, 101, 110, 102, 101, 58, 32, 69, 68, 71, 32, 52, 46, 57
 .b8 0, 4, 99, 97, 108, 108, 49, 46, 99, 117, 0
 .b64 0
 .b32 .debug_line // the .debug_line section will be created by ptxas from the .loc
 .b8 47, 104, 111, 109, 101, 47, 109, 109, 117, 114, 112, 104, 121, 47, 116
 .b8 101, 115, 116, 0, 2, 95, 90, 51, 102, 111, 111, 105, 105, 0, 95, 90
 .b8 51, 102, 111, 111, 105, 105, 0
 .b32 1, 1, 164
 .b8 1
 .b64 func_begin0 // start and end location of foo
 .b64 func_end0
 .b8 1, 156, 3, 105, 0
 .b32 1, 1, 164
 .b8 5, 144, 177, 228, 149, 1, 2, 3, 106, 0
 .b32 1, 1, 164
 .b8 5, 144, 178, 228, 149, 1, 2, 0, 4, 105, 110, 116, 0, 5
 .b32 4
 .b8 2, 95, 90, 52, 116, 101, 115, 116, 80, 105, 0, 95, 90, 52, 116, 101
 .b8 115, 116, 80, 105, 0
 .b32 1, 6, 253
 .b8 1
 .b64 func_begin1 // start and end location of test
 .b64 func_end1
 .b8 1, 156, 3, 112, 0
 .b32 1, 6, 259
 .b8 9, 3
 .b64 _Z4testPi_param_0
 .b8 7, 0, 5, 118, 111, 105, 100, 0, 6
 .b32 164
 .b8 12, 0
}
.section .debug_abbrev {
 .b8 1, 17, 1, 37, 8, 19, 11, 3, 8, 17, 1, 16, 6, 27, 8, 0, 0, 2, 46, 1, 135
 .b8 64, 8, 3, 8, 58, 6, 59, 6, 73, 19, 63, 12, 17, 1, 18, 1, 64, 10, 0, 0
 .b8 3, 5, 0, 3, 8, 58, 6, 59, 6, 73, 19, 2, 10, 51, 11, 0, 0, 4, 36, 0, 3
 .b8 8, 62, 11, 11, 6, 0, 0, 5, 59, 0, 3, 8, 0, 0, 6, 15, 0, 73, 19, 51, 11
 .b8 0, 0, 0
}
.section .debug_pubnames {
 .b32 41
 .b8 2, 0
 .b32 .debug_info
 .b32 262, 69
 .b8 95, 90, 51, 102, 111, 111, 105, 105, 0
 .b32 174
 .b8 95, 90, 52, 116, 101, 115, 116, 80, 105, 0
 .b32 0
}
```
