---
title: "5.3.3. C++17 Language Features"
section: "5.3.3"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#c-17-language-features"
---

## [5.3.3. C++17 Language Features](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#c-17-language-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#c-17-language-features "Permalink to this headline")

| Language Feature | C++17 Proposal | NVCC/CUDA Toolkit 11.x |
| --- | --- | --- |
| Removing trigraphs | [N4086](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html) | ✅ |
| `u8` character literals | [N4267](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html) | ✅ |
| Folding expressions | [N4295](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html) | ✅ |
| Attributes for namespaces and enumerators | [N4266](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html) | ✅ |
| Nested namespace definitions | [N4230](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html) | ✅ |
| Allow constant evaluation for all non-type template arguments | [N4268](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html) | ✅ |
| Extending `static_assert` | [N3928](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf) | ✅ |
| New Rules for `auto` deduction from braced-init-list | [N3922](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html) | ✅ |
| Allow typename in a template template parameter | [N4051](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4051.html) | ✅ |
| `[[fallthrough]]` attribute | [P0188R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf) | ✅ |
| `[[nodiscard]]` attribute | [P0189R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf) | ✅ |
| `[[maybe_unused]]` attribute | [P0212R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf) | ✅ |
| Extension to aggregate initialization | [P0017R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html) | ✅ |
| Wording for `constexpr` lambda | [P0170R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf) | ✅ |
| Unary Folds and Empty Parameter Packs | [P0036R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf) | ✅ |
| Generalizing the Range-Based For Loop | [P0184R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html) | ✅ |
| Lambda capture of `*this` by Value | [P0018R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html) | ✅ |
| Construction Rules for `enum class` variables | [P0138R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf) | ✅ |
| Hexadecimal floating literals for C++ | [P0245R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html) | ✅ |
| Dynamic memory allocation for over-aligned data | [P0035R4](https://wg21.link/p0035) | ✅ |
| Guaranteed copy elision | [P0135R1](https://wg21.link/p0135) | ✅ |
| Refining Expression Evaluation Order for Idiomatic C++ | [P0145R3](https://wg21.link/p0145) | ✅ |
| `constexpr if` | [P0292R2](https://wg21.link/p0292) | ✅ |
| Selection statements with initializer | [P0305R1](https://wg21.link/p0305) | ✅ |
| Template argument deduction for class templates | [P0091R3](https://wg21.link/p0091) <br> [P0512R0](https://wg21.link/p0512r0) | ✅ |
| Declaring non-type template parameters with `auto` | [P0127R2](https://wg21.link/p0127) | ✅ |
| Using attribute namespaces without repetition | [P0028R4](https://wg21.link/p0028) | ✅ |
| Ignoring unsupported non-standard attributes | [P0283R2](https://wg21.link/p0283) | ✅ |
| [Structured bindings](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#structured-binding) | [P0217R3](https://wg21.link/p0217) | ✅ |
| Remove Deprecated Use of the `register` Keyword | [P0001R1](https://wg21.link/p0001) | ✅ |
| Remove Deprecated `operator++(bool)` | [P0002R1](https://wg21.link/p0002) | ✅ |
| Make exception specifications be part of the type system | [P0012R1](https://wg21.link/p0012) | ✅ |
| `__has_include` for C++17 | [P0061R1](https://wg21.link/p0061) | ✅ |
| Rewording inheriting constructors (core issue 1941 et al) | [P0136R1](https://wg21.link/p0136) | ✅ |
| [Inline variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#inline-variables) | [P0386R2](https://wg21.link/p0386r2) | ✅ |
| DR 150, Matching of template template arguments | [P0522R0](https://wg21.link/p0522r0) | ✅ |
| Removing dynamic exception specifications | [P0003R5](https://wg21.link/p0003r5) | ✅ |
| Pack expansions in using-declarations | [P0195R2](https://wg21.link/p0195r2) | ✅ |
| A `byte` type definition | [P0298R0](https://wg21.link/p0298r0) | ✅ |
| DR 727, In-class explicit instantiations | [CWG727](https://cplusplus.github.io/CWG/issues/727.html) | ✅ |
