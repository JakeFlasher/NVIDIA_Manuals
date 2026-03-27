---
title: "Behavioral subtyping"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#behavioral-subtyping"
---

##### [Behavioral subtyping](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#behavioral-subtyping)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#behavioral-subtyping "Permalink to this headline")

Another reason to avoid public inheritance would be if the public member functions of `PipelineA` and `PipelineB` have different behavior, such that the invariants satisfied by the member functions of the base class `PipelineA` are not satisfied by the correspondingly named member functions of the subclass `PipelineB`.  For example, suppose that both classes have a public `producer_arrive` member function.  However, for `PipelineA`, this issues a producer arrival only for its own block, whereas for `PipelineB`, this issues a producer arrival for all blocks in the cluster.  Again, PipelineB “is-not-a” PipelineA.  The child class doesn’t just add behavior onto the parent class; it has completely different behavior. Thus, it fails to satisfy behavioral subtyping: invariants of the parent class’s member functions are not satisfied by the child class.  Behavioral subtyping is especially important when reasoning about already difficult things like parallel synchronization.  The inheritance design would give developers the false impression that `PipelineB` just adds behavior atop `PipelineA`, whereas in fact, developers would need to understand both pipeline classes completely to build a correct mental model about their behavior.

The fix is the same: Use composition, not inheritance.  As [C++ Core Guidelines C.120](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c120-use-class-hierarchies-to-represent-concepts-with-inherent-hierarchical-structure-only) explains: “Use class hierarchies to represent concepts with inherent hierarchical structure (only).”

1. “Make sure the idea represented in the base class exactly matches all derived types and there is not a better way to express it than using the tight coupling of inheritance.”
2. “Do not use inheritance when simply having a data member will do.”
