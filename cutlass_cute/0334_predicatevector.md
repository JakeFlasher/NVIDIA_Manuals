---
title: "PredicateVector#"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html#predicatevector"
---

### [PredicateVector#](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#predicatevector)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#predicatevector "Permalink to this headline")

`PredicateVector<int Bits>` contains a statically sized array of hardware predicates packed into registers to enable efficient access within unrolled loops.

This container is optimized for sequential access through iterators, though these are only efficient when used within fully unrolled loops.

Moreover, instances of `PredicateVector<>` are not guaranteed to be updated until any non-const iterator objects have gone out of scope. This is because iterators are effectively caches that update the `PredicateVector<>` instance’s internal storage as a batch.

**Example:** Managing an array of predicates.

```c++
unsigned mask;
PredicateVector<kBits> predicates;

// Nested scope to update predicates via an iterator
{
  auto pred_it = predicates.begin();

  CUTLASS_PRAGMA_UNROLL
  for (int bit = 0; bit < kBits; ++bit, ++pred_it) {
    bool guard = (mask & (1u << bit));
    pred_it.set(guard);
  }
}

// Efficient use of predicates to guard memory instructions
T *ptr;
Array<T, kAccesses> fragment;

auto pred_it = predicates.const_begin();
for (int access = 0; access < kAccesses; ++access, ++pred_it) {
  if (*pred_it) {
    fragment[access] = ptr[access];
  }
}
```

Note: `PredicateVector<>` is not efficient when accessed via dynamic random access. If an array of bits is needed with dynamic random access (in contrast with access via _constexpr_ indices), then `Array<bin1_t, N>` should be used instead.
