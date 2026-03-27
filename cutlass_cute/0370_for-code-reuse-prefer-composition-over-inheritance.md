---
title: "For code reuse, prefer composition over inheritance"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#for-code-reuse-prefer-composition-over-inheritance"
---

#### [For code reuse, prefer composition over inheritance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#for-code-reuse-prefer-composition-over-inheritance)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#for-code-reuse-prefer-composition-over-inheritance "Permalink to this headline")

- [C++ Core Guidelines C.129](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c129-when-designing-a-class-hierarchy-distinguish-between-implementation-inheritance-and-interface-inheritance): “When designing a class hierarchy, distinguish between implementation inheritance and interface inheritance”
- [C++ Core Guidelines ES.63](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-slice): “Don’t slice”

Suppose that a class hierarchy exists entirely for implementation convenience, so that implementers can reuse code and “program by difference” (changing or adding only what’s different from the base class).  In the example below, both `PipelineA` and `PipelineB` are used by themselves.  `PipelineB` inherits from `PipelineA` just to avoid duplicating code.  There are no virtual member functions, and users don’t expect to rely on run-time polymorphism.

```c++
class PipelineA {
public:
  PipelineA(Arg0 arg0, Arg1 arg1)
    : arg0_(arg0), arg1_(arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    // ... implementation ...
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... implementation ...
  }

private:
  Arg0 arg0_;
  Arg1 arg1_;
};

class PipelineB : public PipelineA {
public:
  PipelineB(Arg0 arg0, Arg1 arg1, Arg2 arg2) :
    PipelineA(arg0, arg1), arg2_(arg2)
  {}

  // Reuse PipelineA::producer_acquire via inheritance

  // Override PipelineA::consumer_release
  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... some other implementation, not invoking parent ...
  }

private:
  Arg2 arg2_;
};
```

The problem with public inheritance here is that `PipelineB` is NOT a (versus “is-a,” i.e., substitutable-as) `PipelineA`. In particular, the following code would be incorrect.

```c++
void consume_and_release_pipeline(PipelineA* parent) {
  // ... code ...
  parent->consumer_release(stage, skip);
  // ... code ...
}

void use_pipeline( /* other args */ ) {
  // ... code ...
  PipelineB child{arg0, arg1, arg2};
  // ... code ...

  // WRONG!!! SLICES CHILD TO PARENT!!!
  consume_and_release_pipeline(&child); // BAD

  // ... code ...
}
```

`PipelineA::consumer_release` is not a virtual member function, so `consume_and_release_pipeline` would not actually be polymorphic, as callers might have expected from an interface that takes a base class pointer. What’s worse is that the resulting slicing could violate `PipelineB`’s invariants, thus putting it in an incorrect state.

The most straightforward way to reuse code would be by changing from inheritance (is-a) to composition (has-a).

```c++
namespace detail {

// Implementation class; not for users
class PipelineImpl {
public:
  PipelineImpl(Arg0 arg0, Arg1 arg1)
    : arg0_(arg0), arg1_(arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    // ... implementation ...
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // ... implementation ...
  }

private:
  Arg0 arg0_;
  Arg1 arg1_;
};

} // namespace detail

class PipelineA {
public:
  PipelineA(Arg0 arg0, Arg1 arg1) :
    impl_(arg0, arg1)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    impl_.producer_acquire(stage, phase, skip_wait);
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    impl_.consumer_release(stage, skip);
  }

private:
  detail::PipelineImpl impl_;
};

// A second kind of pipeline.
// Note that this does NOT inherit from PipelineB!
// The two pipeline classes have the same compile-time interface
// (for compile-time polymorphism), but do not belong in an
// inheritance hierarchy (as would imply run-time polymorphism).
class PipelineB {
public:
  PipelineB(Arg0 arg0, Arg1 arg1, Arg2 arg2) :
    impl_(arg0, arg1), otherTwo_(arg2)
  {}

  void producer_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    impl_.producer_acquire(stage, phase, skip_wait);
  }

  void consumer_release(uint32_t stage, uint32_t skip) {
    // this class doesn't actually use impl_ here
    otherTwo_.other_action(stage, skip);
    // ... some other code not using impl_ ...
  }

private:
  detail::PipelineImpl impl_;
  OtherTwo otherTwo_;
  // ... other member data ...
};
```

This design prevents users at compile time from incorrectly assuming that `PipelineB` is a `PipelineA`.  Implementers continue to get compile-time polymorphism, as long as `PipelineA` and `PipelineB` implement the same compile-time interface.
