---
title: "CUTLASS Utilities"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html#code_organization--cutlass-utilities"
---

### [CUTLASS Utilities](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#cutlass-utilities)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#cutlass-utilities "Permalink to this headline")

`tools/util/` defines a companion library of headers and sources that support the CUTLASS test programs, examples, and other client applications. Its structure is as follows:

```console
tools/
  util/
    include/
      cutlass/
        util/                   # CUTLASS Utility companion library

          reference/            #  functional reference implementation of CUTLASS operators
                                #    (minimal consideration for performance)

            detail/
              *

            device/             #  device-side reference implementations of CUTLASS operators
              thread/
              kernel/
                *
            host/               #  host-side reference implementations of CUTLASS operators
              *
          *
```

[More details about CUTLASS Utilities may be found here.](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html)
