---
title: "General"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html#general"
---

## [General](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#general)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#general "Permalink to this headline")

**Are the DSLs replacing C++ templates?**

> TL;DR: No - but also yes. The CUTLASS 4.0 release (CuTe DSL), along with all
> future extensions to our Python-native programming models, does not come at the
> expense of CUTLASS C++.  CUTLASS 2.x and 3.x C++ APIs are both going to continue
> receiving fixes and updates for the architectures we support them for. However,
> CUTLASS 4.x CuTe DSL is fully isomorphic in its programming model and performance
> with CuTe C++ for Blackwell, and it is our hope that the community embraces this
> for much easier while still equally performant custom kernel development.  This is
> why we are releasing CuTe DSL with support for all architectures starting with the
> NVIDIA Ampere Architecture.

**What is the difference between CuTe DSL, CUTLASS Python, and CUTLASS DSLs?**

> CUTLASS Python was the Python interface for instantiating C++ kernels via a Python
> frontend. This is now deprecated with the release of CUTLASS 4.0. CUTLASS DSLs are
> a family of Python DSLs for native device programming in Python. Currently, this is
> limited to our initial release of CuTe DSL, but future versions will include higher-level
> abstractions that gradually trade off control for convenience.

**What should I learn, CUTLASS C++ or the Python DSLs?**

> We believe the Python DSLs will significantly improve the learning curve and recommend starting
> with them for all newcomers, as they eliminate the inherent complexity of learning C++
> metaprogramming for GPU kernel programming. Since CuTe C++ and CuTe DSL share fully isomorphic
> programming models and patterns, any knowledge gained can eventually be applied to C++.

**Where will the code live? PIP wheel or GitHub repo? Do I have to build it myself?**

> This is a major change compared to CUTLASS C++ and Python DSLs. Going forward,
> the GitHub code only exists as a way for users to file issues and pull requests against.
> While it can be used with the pip wheel, we do not recommend most users do so unless they are
> hacking on the DSL itself. For all other users, we recommend they
> simply `pip install nvidia-cutlass-dsl` and use the pip wheel as the single source
> of truth for the dialect compiler and DSL implementation. CUTLASS GitHub repository will
> contain a `requirements.txt` file pinning the version of the wheel consistent with the state
> of the OSS repository (please see [Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)). This means getting started with
> CUTLASS is easier than ever: no more CMake command lines to learn and no more builds to kick
> off. Simply install the pip wheel and start running the examples.
