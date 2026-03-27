---
title: "5.3.7.7. Argument Dependent Lookup (ADL)"
section: "5.3.7.7"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#argument-dependent-lookup-adl"
---

### [5.3.7.7. Argument Dependent Lookup (ADL)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#argument-dependent-lookup-adl)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#argument-dependent-lookup-adl "Permalink to this headline")

As previously mentioned, the CUDA compiler replaces an extended lambda expression with a placeholder type before invoking the host compiler. One template argument of the placeholder type uses the address of the function that encloses the original lambda expression. This may cause additional namespaces to participate in [Argument-Dependent Lookup (ADL)](https://en.cppreference.com/w/cpp/language/adl.html) for any host function call whose argument types involve the closure type of the extended lambda expression. Consequently, an incorrect function may be selected by the host compiler.

Example:

```cuda
namespace N1 {

struct MyStruct {};

template <typename T>
void my_function(T);

}; // namespace N1

namespace N2 {

template <typename T>
int my_function(T);

template <typename T>
void run(T in) { my_function(in); }

} // namespace N2

void bar(N1::MyStruct in) {
    // For extended device-only lambda, the code sent to the host compiler is replaced with
    // the placeholder type instantiation expression
    //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
    //
    // As a result, the namespace 'N1' participates in ADL lookup of the
    // call to "my_function()" in the body of N2::run, causing ambiguity.
    auto lambda1 = [=] __device__ { };
    N2::run(lambda1);
}
```

In the above example, the CUDA compiler replaced the extended lambda with a placeholder type involving the `N1` namespace. Consequently, the `N1` namespace participates in the ADL lookup for `my_function(in)` in the body of `N2::run()`, resulting in a host compilation failure due to the discovery of multiple overload candidates: `N1::my_function` and `N2::my_function`.
