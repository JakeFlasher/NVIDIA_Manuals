---
title: "6.8.3. Tile Block Execution"
section: "6.8.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#tile-block-execution"
---

### [6.8.3. Tile Block Execution](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-block-execution)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-block-execution "Permalink to this headline")

Execution of a single tile block is isolated from other tile blocks. Tile blocks can only
observe effects of other blocks via global memory which can be used to implement forms
of cooperation or communication.

Function bodies are a series of static-single-assignment (SSA) statements which assign
the result of each operations to a unique variable. Each variable is mapped to a register
in the abstract machine’s register file. A function body is executes statements sequentially, in order.
The compiler is free to reorder statements as long as there is no effect on the program visible
effects or violated program semantics.

For more detailed example programs and explanations of their execution see
[Programming Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#section-prog-model) or [Appendix](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#section-appendix).

The one unique semantic of **Tile IR** is the partitioning of memory operations into _program
ordered_ and _token ordered operations_. All memory operations produce their _result
values_ immediately but the order in which order these operations effect memory is more subtle.

For program ordered operations, the order between any pair memory operations acting on the
same address is defined by the operation’s position in program. Intuitively the effect of
all prior memory operations on the same address will be visible to all subsequent memory
operations on the same address.

In contrast, the order between any pair of token ordered operations is undefined, and
has no relation to program order. The order of a pair of any two token ordered operations
(\(\(A\)\) and \(\(B\)\)) is only defined if established by a direct or transitive relationship
between \(\(A\)\)’s output token and \(\(B\)\)’s input token.

This choice importantly allows a producer of **Tile IR** to induce different memory
ordering semantics by inserting the appropriate memory ordering tokens.

For example starting with a single fresh token depending on by the first operation
with the result token of the first operation being depended upon by the second operation
and so on threading the tokens through each operation in program order.

Token threading like this establishes a ordering of the memory operations which is
consistent with the same program with each token ordered operations being replaced
by a program ordered memory operation.

For a detailed discussion of the memory model and memory operations see [Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model).
