---
title: "5.3.4. C++20 Language Features"
section: "5.3.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#c-20-language-features"
---

## [5.3.4. C++20 Language Features](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#c-20-language-features)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#c-20-language-features "Permalink to this headline")

GCC version ≥ 10.0,  Clang version ≥ 10.0, Microsoft Visual Studio ≥ 2022, and nvc++ version ≥ 20.7.

| Language Feature | C++20 Proposal | NVCC/CUDA Toolkit 12.x |
| --- | --- | --- |
| Default member initializers for bit-fields | [P0683R1](https://wg21.link/p0683r1) | ✅ |
| Fixing `const`-qualified pointers to members | [P0704R1](https://wg21.link/p0704r1) | ✅ |
| Allow lambda capture `[=, this]` | [P0409R2](https://wg21.link/p0409r2) | ✅ |
| `__VA_OPT__` for preprocessor comma elision | [P0306R4](https://wg21.link/p0306r4) <br> [P1042R1](https://wg21.link/p1042r1) | ✅ |
| Designated initializers | [P0329R4](https://wg21.link/p0329r4) | ✅ |
| Familiar template syntax for generic lambdas | [P0428R2](https://wg21.link/p0428r2) | ✅ |
| List deduction of vector | [P0702R1](https://wg21.link/p0702r1) | ✅ |
| Concepts | [P0734R0](https://wg21.link/p0734r0) <br> [P0857R0](https://wg21.link/p0857r0) <br> [P1084R2](https://wg21.link/p1084r2) <br> [P1141R2](https://wg21.link/p1141r2) <br> [P0848R3](https://wg21.link/p0848r3) <br> [P1616R1](https://wg21.link/p1616r1) <br> [P1452R2](https://wg21.link/p1452r2) <br> [P1972R0](https://wg21.link/p1972r0) <br> [P1980R0](https://wg21.link/p1980r0) <br> [P2092R0](https://wg21.link/p2092r0) <br> [P2103R0](https://wg21.link/p2103r0) <br> [P2113R0](https://wg21.link/p2113r0) | ✅ |
| Range-based for statements with initializer | [P0614R1](https://wg21.link/p0614r1) | ✅ |
| Simplifying implicit lambda capture | [P0588R1](https://wg21.link/p0588r1) | ✅ |
| ADL and function templates that are not visible | [P0846R0](https://wg21.link/p0846r0) | ✅ |
| `const` mismatch with defaulted copy constructor | [P0641R2](https://wg21.link/p0641r2) | ✅ |
| Less eager instantiation of `constexpr` functions | [P0859R0](https://wg21.link/p0859r0) | ✅ |
| [Consistent comparison](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cpp20-spaceship) (`operator<=>`) | [P0515R3](https://wg21.link/p0515r3) <br> [P0905R1](https://wg21.link/p0905r1) <br> [P1120R0](https://wg21.link/p1120r0) <br> [P1185R2](https://wg21.link/p1185r2) <br> [P1186R3](https://wg21.link/p1186r3) <br> [P1630R1](https://wg21.link/p1630r1) <br> [P1946R0](https://wg21.link/p1946r0) <br> [P1959R0](https://wg21.link/p1959r0) <br> [P2002R1](https://wg21.link/p2002r1) <br> [P2085R0](https://wg21.link/p2085r0) | ✅ |
| Access checking on specializations | [P0692R1](https://wg21.link/p0692r1) | ✅ |
| Default constructible and assignable stateless lambdas | [P0624R2](https://wg21.link/p0624r2) | ✅ |
| Lambdas in unevaluated contexts | [P0315R4](https://wg21.link/p0315r4) | ✅ |
| Language support for empty objects | [P0840R2](https://wg21.link/p0840r2) | ✅ |
| Relaxing the range-for loop customization point finding rules | [P0962R1](https://wg21.link/p0962r1) | ✅ |
| [Allow structured bindings to accessible members](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#structured-binding) | [P0969R0](https://wg21.link/p0969r0) | ✅ |
| Relaxing the structured bindings customization point finding rules | [P0961R1](https://wg21.link/p0961r1) | ✅ |
| Down with typename! | [P0634R3](https://wg21.link/p0634r3) | ✅ |
| Allow pack expansion in lambda init-capture | [P0780R2](https://wg21.link/p0780r2) <br> [P2095R0](https://wg21.link/p2095r0) | ✅ |
| Proposed wording for `likely` and `unlikely` attributes | [P0479R5](https://wg21.link/p0479r5) | ✅ |
| Deprecate implicit capture of this via `[=]` | [P0806R2](https://wg21.link/p0806r2) | ✅ |
| Class Types in Non-Type Template Parameters | [P0732R2](https://wg21.link/p0732r2) | ✅ |
| Inconsistencies with non-type template parameters | [P1907R1](https://wg21.link/p1907r1) | ✅ |
| Atomic Compare-and-Exchange with Padding Bits | [P0528R3](https://wg21.link/p0528r3) | ✅ |
| Efficient sized delete for variable sized classes | [P0722R3](https://wg21.link/p0722r3) | ✅ |
| Allowing Virtual Function Calls in Constant Expressions | [P1064R0](https://wg21.link/p1064r0) | ✅ |
| Prohibit aggregates with user-declared constructors | [P1008R1](https://wg21.link/p1008r1) | ✅ |
| `explicit(bool)` | [P0892R2](https://wg21.link/p0892r2) | ✅ |
| Signed integers are two’s complement | [P1236R1](https://wg21.link/p1236r1) | ✅ |
| `char8_t` | [P0482R6](https://wg21.link/p0482r6) | ✅ |
| [Immediate functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#cpp20-consteval) (`consteval`) | [P1073R3](https://wg21.link/p1073r3) <br> [P1937R2](https://wg21.link/p1937r2) | ✅ |
| `std::is_constant_evaluated` | [P0595R2](https://wg21.link/p0595r2) | ✅ |
| Nested `inline` namespaces | [P1094R2](https://wg21.link/p1094r2) | ✅ |
| Relaxations of `constexpr` restrictions | [P1002R1](https://wg21.link/p1002r1) <br> [P1327R1](https://wg21.link/p1327r1) <br> [P1330R0](https://wg21.link/p1330r0) <br> [P1331R2](https://wg21.link/p1331r2) <br> [P1668R1](https://wg21.link/p1668r1) <br> [P0784R7](https://wg21.link/p0784r7) | ✅ |
| Feature test macros | [P0941R2](https://wg21.link/p0941r2) | ✅ |
| Modules | [P1103R3](https://wg21.link/p1103r3) <br> [P1766R1](https://wg21.link/p1766r1) <br> [P1811R0](https://wg21.link/p1811r0) <br> [P1703R1](https://wg21.link/p1703r1) <br> [P1874R1](https://wg21.link/p1874r1) <br> [P1979R0](https://wg21.link/p1979r0) <br> [P1779R3](https://wg21.link/p1779r3) <br> [P1857R3](https://wg21.link/p1857r3) <br> [P2115R0](https://wg21.link/p2115r0) <br> [P1815R2](https://wg21.link/p1815r2) | ❌ |
| Coroutines | [P0912R5](https://wg21.link/p0912r5) | ❌ |
| Parenthesized initialization of aggregates | [P0960R3](https://wg21.link/p0960r3) <br> [P1975R0](https://wg21.link/p1975r0) | ✅ |
| DR: array size deduction in new-expression | [P1009R2](https://wg21.link/p1009r2) | ✅ |
| DR: Converting from `T*` to bool should be considered narrowing | [P1957R2](https://wg21.link/p1957r2) | ✅ |
| Stronger Unicode requirements | [P1041R4](https://wg21.link/p1041r4) <br> [P1139R2](https://wg21.link/p1139r2) | ✅ |
| Structured binding extensions | [P1091R3](https://wg21.link/p1091r3) <br> [P1381R1](https://wg21.link/p1381r1) | ✅ |
| Deprecate `a[b,c]` | [P1161R3](https://wg21.link/p1161r3) | ✅ |
| Deprecating some uses of `volatile` | [P1152R4](https://wg21.link/p1152r4) | ✅ |
| `[[nodiscard("with reason")]]` | [P1301R4](https://wg21.link/p1301r4) | ✅ |
| `using enum` | [P1099R5](https://wg21.link/p1099r5) | ✅ |
| Class template argument deduction for aggregates | [P1816R0](https://wg21.link/p1816r0) <br> [P2082R1](https://wg21.link/p2082r1) | ✅ |
| Class template argument deduction for alias templates | [P1814R0](https://wg21.link/p1814r0) | ✅ |
| Permit conversions to arrays of unknown bound | [P0388R4](https://wg21.link/p0388r4) | ✅ |
| `constinit` | [P1143R2](https://wg21.link/p1143r2) | ✅ |
| Layout-compatibility and Pointer-interconvertibility Traits | [P0466R5](https://wg21.link/p0466r5) | ✅ |
| DR: Checking for abstract class types | [P0929R2](https://wg21.link/p0929r2) | ✅ |
| DR: More implicit moves | [P1825R0](https://wg21.link/p1825r0) | ✅ |
| DR: Pseudo-destructors end object lifetimes | [P0593R6](https://wg21.link/p0593r6) | ✅ |
