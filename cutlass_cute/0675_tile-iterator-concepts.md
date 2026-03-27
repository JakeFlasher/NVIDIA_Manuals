---
title: "Tile Iterator Concepts"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/tile_iterator_concept.html#tile-iterator-concepts"
---

# [Tile Iterator Concepts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#tile-iterator-concepts)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#tile-iterator-concepts "Permalink to this headline")

Note: CUTLASS 3.0 deprecates all tile access iterators in favour of CuTe’s single
vocabulary type `cute::Tensor`, which is parameterized on `cute::Layout`.
`cute::Tensor`s can therefore be manipulated with the same layout algebra as all CuTe layouts.
This removes the need for bespoke types that encapsulate iterator properties.
The following text thus only applies to legacy CUTLASS 2.x API and related types.

CUTLASS 2.x implements generic algorithms on tiles of matrix or tensors of constant size. These may
be considered as partitions of tensors of infinite size, with a range of partitions accessible
by _tile iterators_.

Various data structures may make operations such as random access to tiles inexpensive,
while data structures may not offer random access at all. For example, iterating over a linked
list of matrices requires sequential traversal. Algorithms implemented in terms of sequences of tiles
should require only the minimum set of operators be defined for tile iterators.

This document describes a set of C++ concepts which may be used to define tile iterators used
by CUTLASS algorithms.  (“Concept” here does not refer to a C++20 concept that uses the `concept` keyword.
Rather, it refers to a set of requirements on a type.)
Each concept specifies members and type definitions that a tile iterator
must implement. Frequently, a tile iterator implements several concepts, and its members are
the union of the members from each individual concept. These definitions were inspired by
[Boost “New style” iterator concepts](https://www.boost.org/doc/libs/1_40_0/libs/iterator/doc/new-iter-concepts.html).

The set of all possible combinations of these concepts is quite large, however most tile iterator
templates can be described by one of several combinations. The section
Frequently Used Tile Iterator Concepts describes several common interfaces used throughout CUTLASS.
