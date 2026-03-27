---
title: "10.3. Feature Availability & Emulation"
section: "10.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#feature-availability-emulation"
---

## [10.3. Feature Availability & Emulation](https://docs.nvidia.com/cuda/tile-ir/latest/sections#feature-availability-emulation)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#feature-availability-emulation "Permalink to this headline")

**Target-specific Features**
**Tile IR** may introduce new target-specific features (e.g., new datatypes, new operations) over time.

- **Availability**: A feature introduced in vX.Y becomes usable on a hardware target starting with the first platform release that declares support for it.
- **Fallback**: If a program uses a feature unsupported by the selected hardware target, the compiler will either diagnose the incompatibility or apply a lowering (emulation) that preserves semantics as defined by the specification.

Note that certain types have more restricted usage than others. See [Element Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subsection-element-types) for details.

> **Warning**
>
> During the 13.x release cycle, we are bringing up existing hardware targets which may introduce new features on old targets. This “cold start” period is an exception; normally, new features will only appear in new targets.

> **Note**
>
> Today the only target-specific features are specific datatypes.

| Data Type | Ampere | Ada | Hopper | Blackwell |
| --- | --- | --- | --- | --- |
| i1, i8, i16, i32, i64 | Supported | Supported | n/a | Supported |
| f16, f32, f64 | Supported | Supported | n/a | Supported |
| bf16, tf32 | Supported | Supported | n/a | Supported |
| f8E3M4, f8E5M2 | Not Supported | Not Supported | n/a | Supported |

**Emulation**

To maintain portability, **Tile IR** may emulate operations on hardware targets that lack native support. For example, 16-bit
floating-point operations may be emulated using 32-bit instructions if the target does not support them natively.
