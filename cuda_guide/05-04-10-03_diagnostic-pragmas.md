---
title: "5.4.10.3. Diagnostic Pragmas"
section: "5.4.10.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#diagnostic-pragmas"
---

### [5.4.10.3. Diagnostic Pragmas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#diagnostic-pragmas)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#diagnostic-pragmas "Permalink to this headline")

The following pragmas can be used to manage the severity of errors that are triggered when a specific diagnostic message is raised.

```cuda
#pragma nv_diag_suppress
#pragma nv_diag_warning
#pragma nv_diag_error
#pragma nv_diag_default
#pragma nv_diag_once
```

The uses of these pragmas are as follows:

```cuda
#pragma nv_diag_xxx <error_number1>, <error_number2> ...
```

The affected diagnostic is specified using the error number shown in the warning message. Any diagnostic can be changed to an error, but only warnings can have their severity suppressed or restored after being changed to an error. The `nv_diag_default` pragma returns the severity of a diagnostic to the severity that was in effect before any other pragmas were issued, namely, the normal severity of the message as modified by any command-line options. The following example suppresses the `declared but never referenced` warning of `foo()`:

```cuda
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diag_default 177
void bar() {
    int i = 0;
}
```

The following pragmas may be used to save and restore the current diagnostic pragma state:

```cuda
#pragma nv_diagnostic push
#pragma nv_diagnostic pop
```

Examples:

```cuda
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diagnostic pop
void bar() {
    int i = 0; // raise a warning
}
```

Note that these directives only affect the `nvcc` CUDA front-end compiler. They have no effect on the host compiler.

`nvcc` defines the macro `__NVCC_DIAG_PRAGMA_SUPPORT__` when diagnostic pragmas are supported.
