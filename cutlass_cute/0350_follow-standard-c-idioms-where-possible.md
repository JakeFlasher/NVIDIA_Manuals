---
title: "Follow Standard C++ idioms where possible"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#follow-standard-c-idioms-where-possible"
---

#### [Follow Standard C++ idioms where possible](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#follow-standard-c-idioms-where-possible)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#follow-standard-c-idioms-where-possible "Permalink to this headline")

Regarding “standard C++ idioms,”
CUTLASS source code follows the following guidelines,
with deviations only because of compiler limitations
or where performance absolutely requires it.
“Performance requires it” implies measurement.
Deviations should be limited in scope
and we should always strive to eliminate them.

- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
