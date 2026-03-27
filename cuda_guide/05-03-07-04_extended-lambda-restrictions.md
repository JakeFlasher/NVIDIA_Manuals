---
title: "5.3.7.4. Extended Lambda Restrictions"
section: "5.3.7.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambda-restrictions"
---

### [5.3.7.4. Extended Lambda Restrictions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#extended-lambda-restrictions)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#extended-lambda-restrictions "Permalink to this headline")

Before invoking the host compiler, the CUDA compiler replaces an extended lambda expression with an instance of a placeholder type defined in namespace scope. The placeholder type’s template argument requires taking the address of a function that encloses the original extended lambda expression. This is necessary for correctly executing any `__global__` function template whose template argument involves the closure type of an extended lambda. The enclosing function is computed as follows.

By definition, an extended lambda is present within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

- If the function is not the `operator()` of a lambda expression, it is considered the enclosing function for the extended lambda.
- Otherwise, the extended lambda is defined within the immediate or nested block scope of the `operator()` of one or more enclosing lambda expressions.
  - If the outermost lambda expression is defined within the immediate or nested block scope of a function `F`, then `F` is the computed enclosing function.
  - Otherwise, the enclosing function does not exist.

Example:

```cuda
void host_function() {
    auto lambda1 = [] __device__ { }; // enclosing function for lambda1 is "host_function()"
    auto lambda2 = [] {
        auto lambda3 = [] {
            auto lambda4 = [] __host__ __device__ { }; // enclosing function for lambda4 is "host_function"
        };
    };
}

auto global_lambda = [] {
    auto lambda5 = [] __host__ __device__ { }; // enclosing function for lambda5 does not exist
};
```

---

**Extended Lambda Restrictions**

1. An extended lambda cannot be defined inside another extended lambda expression. Example:

```cuda
void host_function() {
    auto lambda1 = [] __host__ __device__  {
         // ERROR, extended lambda defined within another extended lambda
        auto lambda2 = [] __host__ __device__ { };
    };
}
```
2. An extended lambda cannot be defined inside a generic lambda expression. Example:

```cuda
void host_function() {
    auto lambda1 = [] (auto) {
         // ERROR, extended lambda defined within a generic lambda
        auto lambda2 = [] __host__ __device__ { };
    };
}
```
3. If an extended lambda is defined within the immediate or nested block scope of one or more nested lambda expressions, then the outermost lambda expression must be defined within the immediate or nested block scope of a function. Example:

```cuda
auto lambda1 = []  {
    // ERROR, outer enclosing lambda is not defined within a non-lambda-operator() function
    auto lambda2 = [] __host__ __device__ { };
};
```
4. The enclosing function of the extended lambda must be named, and its address must be accessible. If the enclosing function is a class member, the following conditions must be met:

Example:

```cuda
void host_function() {
    auto lambda1 = [] __device__ { return 0; }; // OK
    {
        auto lambda2 = [] __device__          { return 0; }; // OK
        auto lambda3 = [] __device__ __host__ { return 0; }; // OK
    }
}
struct MyStruct1 {
    MyStruct1() {
        auto lambda4 = [] __device__ { return 0; }; // ERROR, address of the enclosing function is not accessible
    }
};
class MyStruct2 {
    void foo() {
        auto temp1 = [] __device__ { return 10; }; // ERROR, enclosing function has private access in parent class
    }
    struct MyStruct3 {
        void foo() {
            auto temp1 = [] __device__ { return 10; };  // ERROR, enclosing class MyStruct3 has private access in its parent class
        }
    };
};
```
  - All classes enclosing the member function must have a name.
  - The member function must not have private or protected access within its parent class.
  - All enclosing classes must not have private or protected access within their respective parent classes.
5. At the point where the extended lambda has been defined, it must be possible to unambiguously take the address of the enclosing routine. However, this may not always be feasible, for example, when an alias declaration shadows a template type argument with the same name. Example:

```cuda
template <typename T>
struct A {
    using Bar = void;
    void test();
};
template<>
struct A<void> { };
template <typename Bar>
void A<Bar>::test() {
    // In code sent to host compiler, nvcc will inject an address expression here, of the form:
    //   (void (A< Bar> ::*)(void))(&A::test))
    //  However, the class typedef 'Bar' (to void) shadows the template argument 'Bar',
    //  causing the address expression in A<int>::test to actually refer to:
    //    (void (A< void> ::*)(void))(&A::test))
    //  which doesn't take the address of the enclosing routine 'A<int>::test' correctly.
    auto lambda1 = [] __host__ __device__ { return 4; };
}
int main() {
    A<int> var;
    var.test();
}
```
6. An extended lambda cannot be defined in a class that is local to a function. Example:

```cuda
void host_function() {
    struct MyStruct {
        void bar() {
            // ERROR, bar() is member of a class that is local to a function
            auto lambda2 = [] __host__ __device__ { return 0; };
        }
    };
}
```
7. The enclosing function for an extended lambda cannot have deduced return type. Example:

```cuda
auto host_function() {
    // ERROR, the return type of host_function() is deduced
    auto lambda3 = [] __host__ __device__ { return 0; };
}
```
8. A host-device extended lambda cannot be a generic lambda, namely a lambda with an `auto` parameter type. Example:

```cuda
void host_function() {
    // ERROR, __host__ __device__ extended lambdas cannot be a generic lambda
    auto lambda1 = [] __host__ __device__ (auto i) { return i; };
    // ERROR, a host-device extended lambda cannot be a generic lambda
    auto lambda2 = [] __host__ __device__ (auto... i) {
        return sizeof...(i);
    };
}
```
9. If the enclosing function is an instantiation of a function or member template, or if the function is a member of a class template, then the template(s) must satisfy the following constraints:

Example 1:

```cuda
template <template <typename...> class T,
          typename... P1,
          typename... P2>
void bar1(const T<P1...>, const T<P2...>) {
    // ERROR, enclosing function has multiple parameter packs
    auto lambda = [] __device__ { return 10; };
}
template <template <typename...> class T,
          typename... P1,
          typename    T2>
void bar2(const T<P1...>, T2) {
    // ERROR, for enclosing function, the parameter pack is not last in the template parameter list
    auto lambda = [] __device__ { return 10; };
}
template <typename T, T>
void bar3() {
    // ERROR, for enclosing function, the second template parameter is not named
    auto lambda = [] __device__ { return 10; };
}
```

Example 2:

```cuda
template <typename T>
void bar4() {
    auto lambda1 = [] __device__ { return 10; };
}
class MyStruct {
    struct MyNestedStruct {};
    friend int main();
};
int main() {
    struct MyLocalStruct {};
    // ERROR, enclosing function for device lambda in bar4() is instantiated with a type local to main
    bar4<MyLocalStruct>();
    // ERROR, enclosing function for device lambda in bar4 is instantiated with a type
    //        that is a private member of a class
    bar4<MyStruct::MyNestedStruct>();
}
```
  - The template must have at most one variadic parameter, and it must be listed last in the template parameter list.
  - The template parameters must be named.
  - The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are `private` or `protected` class members.
10. With Microsoft Visual Studio host compilers, the enclosing function must have external linkage. This restriction exists because the host compiler does not support using the addresses of non-extern linkage functions as template arguments. The CUDA compiler transformations require these addresses to support extended lambdas.
11. With Microsoft Visual Studio host compilers, an extended lambda shall not be defined within the body of an `if constexpr` block.
12. An extended lambda has the following restrictions on captured variables:

Examples:

```cuda
void host_function() {
    // CORRECT, an init-capture is allowed for an extended device-only lambda
    auto lambda1 = [x = 1] __device__ () { return x; };
    // ERROR, an init-capture is not allowed for an extended host-device lambda
    auto lambda2 = [x = 1] __host__ __device__ () { return x; };
    int a = 1;
    // ERROR, an extended __device__ lambda cannot capture variables by reference
    auto lambda3 = [&a] __device__ () { return a; };
    // ERROR, by-reference capture is not allowed for an extended device-only lambda
    auto lambda4 = [&x = a] __device__ () { return x; };
    struct MyStruct {};
    MyStruct s1;
    // ERROR, a type local to a function cannot be used in the type of a captured variable
    auto lambda6 = [s1] __device__ () { };
    // ERROR, an init-capture cannot be of type std::initializer_list
    auto lambda7 = [x = {11}] __device__ () { };
    std::initializer_list<int> b = {11,22,33};
    // ERROR, an init-capture cannot be of type std::initializer_list
    auto lambda8 = [x = b] __device__ () { };
    int  var     = 4;
    auto lambda9 = [=] __device__ {
        int result = 0;
        if constexpr(false) {
            //ERROR, An extended device-only lambda cannot first-capture 'var' in if-constexpr context
            result += var;
        }
        return result;
    };
    auto lambda10 = [var] __device__ {
        int result = 0;
        if constexpr(false) {
            // CORRECT, 'var' already listed in explicit capture list for the extended lambda
            result += var;
        }
        return result;
    };
    auto lambda11 = [=] __device__ {
        int result = var;
        if constexpr(false) {
            // CORRECT, 'var' already implicit captured outside the 'if-constexpr' block
            result += var;
        }
        return result;
    };
}
```
  - The variable may be passed by value to a sequence of helper functions in the code sent to the host compiler before being used to directly initialize the field of the class type representing the closure type for the extended lambda. However, the C++ standard specifies that the captured variable should be used for direct initialization of the closure type’s field.
  - A variable can only be captured by value.
  - A variable of array type cannot be captured if the number of array dimensions is greater than 7.
  - For an array-type variable, the array field of the closure type is first default-initialized and then each array element is copy-assigned from the corresponding element of the captured array variable in the code sent to the host compiler. Therefore, the array element type must be both default-constructible and copy-assignable in the host code.
  - A function parameter that is an element of a variadic argument pack cannot be captured.
  - The captured variable type cannot be local to a function, except for extended lambda closure types, or `private` or `protected` class members.
  - Init-capture is not supported for host-device extended lambdas. However, it is supported for device extended lambdas, except when the initializer is an array or of type `std::initializer_list`.
  - The function call operator for an extended lambda is not a `constexpr`. The closure type of an extended lambda is not a literal type. The `constexpr` and `consteval` specifiers cannot be used when declaring an extended lambda.
  - A variable cannot be implicitly captured inside an `if-constexpr` block that is lexically nested inside an extended lambda unless the variable has been implicitly captured outside the `if-constexpr` block or appears in the extended lambda’s explicit capture list.
13. When parsing a function, the CUDA compiler assigns a counter value to each extended lambda in the function. This counter value is used in the substituted named type that is passed to the host compiler. Therefore, the presence or absence of an extended lambda within a function should not depend on a particular value of `__CUDA_ARCH__`, nor on `__CUDA_ARCH__` being undefined. Example:

```cuda
template <typename T>
__global__ void kernel(T in) { in(); }
__host__ __device__ void host_device_function() {
    // ERROR, the number and relative declaration order of
    //        extended lambdas depend on __CUDA_ARCH__
#if defined(__CUDA_ARCH__)
    auto lambda1 = [] __device__ { return 0; };
    auto lambda2 = [] __host__ __device__ { return 10; };
#endif
    auto lambda3 = [] __device__ { return 4; };
    kernel<<<1, 1>>>(lambda3);
}
```
14. As described above, the CUDA compiler replaces a device extended lambda defined in a host function with a placeholder type defined in namespace scope. The placeholder type does not define an `operator()` function equivalent to the original lambda declaration unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true` for the closure type of the extended lambda. Therefore, an attempt to determine the return type or parameter types of the `operator()` function of such a lambda may work incorrectly in host code because the code processed by the host compiler is semantically different from the input code processed by the CUDA compiler. However, introspecting the return type or parameter types of the `operator()` function within device code is acceptable. Note that this restriction does not apply to host or device extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`. Example:

```cuda
#include <cuda/std/type_traits>
const char& getRef(const char* p) { return *p; }
void foo() {
    auto lambda1 = [] __device__ { return "10"; };
    // ERROR, attempt to extract the return type of a device lambda in host code
    cuda::std::result_of<decltype(lambda1)()>::type xx1 = "abc";
    auto lambda2 = [] __host__ __device__ { return "10"; };
    // CORRECT, lambda2 represents a host-device extended lambda
    cuda::std::result_of<decltype(lambda2)()>::type xx2 = "abc";
    auto lambda3 = [] __device__ () -> const char* { return "10"; };
    // CORRECT, lambda3 represents a device extended lambda with preserved return type
    cuda::std::result_of<decltype(lambda3)()>::type xx2 = "abc";
    static_assert(cuda::std::is_same_v<cuda::std::result_of<decltype(lambda3)()>::type, const char*>);
    auto lambda4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
    // lambda4's return type is not preserved because it references the operator()'s
    // parameter types in the trailing return type.
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(decltype(lambda4)));
}
```
15. For an extended device-only lambda:
  - Introspection of the parameter type of `operator()` is only supported in device code.
  - Introspection of the return type of `operator()` is supported only in device code, unless the trait function `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`.
16. If an extended lambda is passed from host to device code as an argument to a `__global__` function, for example, then any expression in the lambda’s body that captures variables must remain unchanged, regardless of whether the `__CUDA_ARCH__` macro is defined and what value it has. This restriction arises because the lambda’s closure class layout depends on the order in which the compiler encounters the captured variables when processing the lambda expression. The program may execute incorrectly if the closure class layout differs between device and host compilations. Example:

```cuda
__device__ int result;
template <typename T>
__global__ void kernel(T in) { result = in(); }
void foo(void) {
    int x1 = 1;
    // ERROR, "x1" is only captured when __CUDA_ARCH__ is defined.
    auto lambda1 = [=] __host__ __device__ {
#ifdef __CUDA_ARCH__
        return x1 + 1;
#else
        return 10;
#endif
    };
    kernel<<<1, 1>>>(lambda1);
}
```
17. As previously described, the CUDA compiler replaces an extended device-only lambda expression with a placeholder type instance in the code sent to the host compiler. The placeholder type does not define a pointer-to-function conversion operator in the host code; however, the conversion operator is provided in the device code. Note that this restriction does not apply to host-device extended lambdas. Example:

```cuda
template <typename T>
__global__ void kernel(T in) {
    int (*fp)(double) = in;
    fp(0); // CORRECT, conversion in device code is supported
    auto lambda1 = [](double) { return 1; };
}
void foo() {
    auto lambda_device      = [] __device__ (double) { return 1; };
    auto lambda_host_device = [] __host__ __device__ (double) { return 1; };
    kernel<<<1, 1>>>(lambda_device);
    kernel<<<1, 1>>>(lambda_host_device);
    // CORRECT, conversion for a __host__ __device__ lambda is supported in host code
    int (*fp1)(double) = lambda_host_device;
    // ERROR, conversion for a device lambda is not supported in host code
    int (*fp2)(double) = lambda_device;
}
```
18. As previously described, the CUDA compiler replaces an extended device-only or host-device lambda expression with a placeholder type instance in the code sent to the host compiler. This placeholder type may define C++ special member functions, such as constructors and destructors. Consequently, some standard C++ type traits may yield different results for the closure type of the extended lambda in the CUDA front-end compiler than in the host compiler. The following type traits are affected: : `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`. Care must be taken to ensure that the results of these traits are not used in the instantiation of the `__global__`, `__device__`, `__constant__`, or `__managed__` function or variable templates. Example:

```cuda
#include <cstdio>
#include <type_traits>
template <bool b>
void __global__ kernel() { printf("hi"); }
template <typename T>
void kernel_launch() {
    // ERROR, this kernel launch may fail, because CUDA frontend compiler and host compiler
    //        may disagree on the result of std::is_trivially_copyable_v trait on the
    //        closure type of the extended lambda
    kernel<std::is_trivially_copyable_v<T>><<<1,1>>>();
    cudaDeviceSynchronize();
}
int main() {
    int  x       = 0;
    auto lambda1 = [=] __host__ __device__ () { return x; };
    kernel_launch<decltype(lambda1)>();
}
```

The CUDA compiler will generate compiler diagnostics for a subset of cases described in `1-12`; no diagnostic will be generated for cases `13-17`, but the host compiler may fail to compile the generated code.
