---
title: "4. System Calls"
section: "4"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#system-calls"
---

# [4. System Calls](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#system-calls)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#system-calls "Permalink to this headline")

System calls are calls into the driver operating system code. In PTX they look like regular calls, but the function definition is not given. A prototype must be provided in the PTX file, but the implementation of the function is provided by the driver.

The prototype for the vprintf system call is:

```c++
.extern .func (.param .s32 status) vprintf (.param t1 format, .param t2 valist)
```

The following are the definitions for the vprintf parameters and return value.

- status : The status value that is returned by vprintf.
- format : A pointer to the format specifier input. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.
- valist : A pointer to the valist input. For 32-bit addresses, type t2 is .b32. For 64-bit addresses, type t2 is .b64.

A call to vprintf using 32-bit addresses looks like:

```c++
cvta.global.b32    %r2, _fmt;
st.param.b32  [param0], %r2;
cvta.local.b32  %r3, _valist_array;
st.param.b32  [param1], %r3;
call.uni (_), vprintf, (param0, param1);
```

For this code, _fmt is the format string in global memory, and _valist_array is the valist of arguments. Note that any pointers must be converted to generic space. The vprintf syscall is emitted as part of the printf function defined in “stdio.h”.

The prototype for the malloc system call is:

```c++
.extern .func (.param t1 ptr) malloc (.param t2 size)
```

The following are the definitions for the malloc parameters and return value.

- ptr : The pointer to the memory that was allocated by malloc. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.
- size : The size of memory needed from malloc. This size is defined by the type size_t. When size_t is 32 bits, type t2 is .b32. When size_t is 64 bits, type t2 is .b64.

The prototype for the free system call is:

```c++
.extern .func free (.param t1 ptr)
```

The following is the definition for the free parameter.

- ptr : The pointer to the memory that should be freed. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.

The malloc and free system calls are emitted as part of the malloc and free functions defined in “malloc.h”.

In order to support assert, the PTX function call __assertfail is used whenever the assert expression produces a false value. The prototype for the __assertfail system call is:

```c++
.extern .func __assertfail (.param t1 message, .param t1 file, .param .b32 line, .param t1 function, .param t2 charSize)
```

The following are the definitions for the __assertfail parameters.

- message : The pointer to the string that should be output. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.
- file : The pointer to the file name string associated with the assert. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.
- line : The line number associated with the assert.
- function : The pointer to the function name string associated with the assert. For 32-bit addresses, type t1 is .b32. For 64-bit addresses, type t1 is .b64.
- charSize : The size in bytes of the characters contained in the __assertfail parameter strings. The only supported character size is 1. The character size is defined by the type size_t. When size_t is 32 bits, type t2 is .b32. When size_t is 64 bits, type t2 is .b64.

The __assertfail system call is emitted as part of the assert macro defined in “assert.h”.
