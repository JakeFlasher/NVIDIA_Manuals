---
title: "Migration"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html#migration"
---

## [Migration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#migration)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#migration "Permalink to this headline")

**Should I port my code from C++ templates to Python?**

> Almost certainly not, unless you need extremely fast JIT times for your kernel and C++ compile times
> are a blocker for you. The 2.x and 3.x APIs will continue to be supported, and Nvidia’s Hopper and
> Blackwell architectures 3.x will continue to improve in terms of features
> and performance.

**Are portability promises different with Python?**

> For the initial release while the DSL is still in beta, we do not promise any portability
> as we may make changes to the DSL itself. While we do not expect any changes to the CuTe operations,
> the DSL utilities, decorators, helper classes like pipelines and schedulers may change as we refine them
> with community feedback. We encourage users to file issues and discussions on GitHub during this
> beta period with their feedback!
>
>
> In the long term, we plan to continue to treat the OSS community with care.
> Just like the prior history of CUTLASS, we plan not to break users unless necessary,
> but we reserve the right to make limited breaking changes in case we believe it is a
> net benefit to the community and project. These will be announced ahead of time and/or
> clearly highlighted in the CHANGELOG of each release.
