---
title: "Reducing the number of named types and iterator concepts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html#reducing-the-number-of-named-types-and-iterator-concepts"
---

## [Reducing the number of named types and iterator concepts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#reducing-the-number-of-named-types-and-iterator-concepts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#reducing-the-number-of-named-types-and-iterator-concepts "Permalink to this headline")

CUTLASS 2.x design preferred introducing bespoke named types for each
architecture specific thread and data layout. For instance, `gemm::treadblock` namespace
contains implementation for `MmaMultistage`, `MmaPlanarComplexMultistage`, `MmaPipelined` etc.
despite them providing mainloops for GEMMs. To spell these types the same way in generic code,
CUTLASS 2.x provides aliases through its `default_x_configuration.h` files, however,
these aliases make the code much harder to read as the user has to perform type substitution
mentally in order to understand the codebase.

CUTLASS 3.0 greatly reduces the number of named types used throughout by

- Replacing all iterator concepts for all memory domains with `cute::Tensor`s
- Dispatching mainloop and epilogue implementations on tag-dispatch policies rather than naming new types
- Dispatching kernel layer schedules on tag-dispatch policies rather than naming new types

Reducing the number of named types has many benefits:

- It _makes writing generic code easier_, as the primary type names share the same lexical
without aliasing through configuration providers.
- It _flattens the learning curve of CUTLASS_ by greatly reducing the mental context required
as the library only exposes a handful of named types.
- It _provides a clear, singular extension point_ for users to plug in their customizations
through the dispatch policies.
