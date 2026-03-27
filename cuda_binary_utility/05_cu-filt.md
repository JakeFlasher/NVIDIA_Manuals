---
title: "5. cu++filt"
section: "5"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#cu-filt"
---

# [5. cu++filt](https://docs.nvidia.com/cuda/cuda-binary-utilities#cu-filt)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cu-filt "Permalink to this headline")

*cu++filt* decodes (demangles) low-level identifiers that have been mangled by CUDA C++ into user readable names. For every input alphanumeric word, the output of `cu++filt` is either the demangled name if the name decodes to a CUDA C++ name, or the original name itself.

## [5.1. Usage](https://docs.nvidia.com/cuda/cuda-binary-utilities#id6)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#id6 "Permalink to this headline")

`cu++filt` accepts one or more alphanumeric words (consisting of letters, digits, underscores, dollars, or periods) and attepts to decipher them. The basic usage is as following:

```text
cu++filt [options] <symbol(s)>
```

To demangle an entire file, like a binary, pipe the contents of the file to cu++filt, such as in the following command:

```text
nm <input file> | cu++filt
```

To demangle function names without printing their parameter types, use the following command :

```text
cu++filt -p <symbol(s)>
```

To skip a leading underscore from mangled symbols, use the following command:

```text
cu++filt -_ <symbol(s)>
```

Here’s a sample output of `cu++filt`:

```text
$ cu++filt _Z1fIiEbl
bool f<int>(long)
```

As shown in the output, the symbol `_Z1fIiEbl` was successfully demangled.

To strip all types in the function signature and parameters, use the `-p` option:

```text
$ cu++filt -p _Z1fIiEbl
f<int>
```

To skip a leading underscore from a mangled symbol, use the `-_` option:

```text
$ cu++filt -_ __Z1fIiEbl
bool f<int>(long)
```

To demangle an entire file, pipe the contents of the file to cu++filt:

```text
$ nm test.cubin | cu++filt
0000000000000000 t hello(char *)
0000000000000070 t hello(char *)::display()
0000000000000000 T hello(int *)
```

Symbols that cannot be demangled are printed back to stdout as is:

```text
$ cu++filt _ZD2
_ZD2
```

Multiple symbols can be demangled from the command line:

```text
$ cu++filt _ZN6Scope15Func1Enez _Z3fooIiPFYneEiEvv _ZD2
Scope1::Func1(__int128, long double, ...)
void foo<int, __int128 (*)(long double), int>()
_ZD2
```

## [5.2. Command-line Options](https://docs.nvidia.com/cuda/cuda-binary-utilities#cuplusplusfilt-options)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuplusplusfilt-options "Permalink to this headline")

[Table 9](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuplusplusfilt-options-table) contains supported command-line options of `cu++filt`, along with a description of what each option does.

| Option | Description |
| --- | --- |
| `-_` | Strip underscore. On some systems, the CUDA compiler puts an underscore in front of every name. This option removes the initial underscore. Whether cu++filt removes the underscore by default is target dependent. |
| `-p` | When demangling the name of a function, do not display the types of the function’s parameters. |
| `-h` | Print a summary of the options to cu++filt and exit. |
| `-v` | Print the version information of this tool. |

## [5.3. Library Availability](https://docs.nvidia.com/cuda/cuda-binary-utilities#library-availability)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#library-availability "Permalink to this headline")

`cu++filt` is also available as a static library (libcufilt) that can be linked against an existing project. The following interface describes it’s usage:

```text
char* __cu_demangle(const char *id, char *output_buffer, size_t *length, int *status)
```

This interface can be found in the file “nv_decode.h” located in the SDK.

**Input Parameters**

_id_ Input mangled string.

_output_buffer_ Pointer to where the demangled buffer will be stored. This memory must be allocated with malloc. If output-buffer is NULL, memory will be malloc’d to store the demangled name and returned through the function return value. If the output-buffer is too small, it is expanded using realloc.

_length_ It is necessary to provide the size of the output buffer if the user is providing pre-allocated memory. This is needed by the demangler in case the size needs to be reallocated. If the length is non-null, the length of the demangled buffer is placed in length.

_status_ *status is set to one of the following values:

> - 0 - The demangling operation succeeded
> - -1 - A memory allocation failure occurred
> - -2 - Not a valid mangled id
> - -3 - An input validation failure has occurred (one or more arguments are invalid)

**Return Value**

A pointer to the start of the NUL-terminated demangled name, or NULL if the demangling fails. The caller is responsible for deallocating this memory using free.

**Note**: This function is thread-safe.

**Example Usage**

```text
#include <stdio.h>
#include <stdlib.h>
#include "nv_decode.h"

int main()
{
  int     status;
  const char *real_mangled_name="_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
  const char *fake_mangled_name="B@d_iDentiFier";

  char* realname = __cu_demangle(fake_mangled_name, 0, 0, &status);
  printf("fake_mangled_name:\t result => %s\t status => %d\n", realname, status);
  free(realname);

  size_t size = sizeof(char)*1000;
  realname = (char*)malloc(size);
  __cu_demangle(real_mangled_name, realname, &size, &status);
  printf("real_mangled_name:\t result => %s\t status => %d\n", realname, status);
  free(realname);

  return 0;
}
```

This prints:

```text
fake_mangled_name:   result => (null)     status => -2
real_mangled_name:   result => clstmp01<cls01>::clstmp01_mf01()   status => 0
```
