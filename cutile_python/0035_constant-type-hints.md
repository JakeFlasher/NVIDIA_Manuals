---
title: "Constant Type Hints"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#constant-type-hints"
---

### [Constant Type Hints](https://docs.nvidia.com/cuda/cutile-python#constant-type-hints)[](https://docs.nvidia.com/cuda/cutile-python/#constant-type-hints "Permalink to this headline")

```python
import cuda.tile as ct
```

```python
def needs_constant(x: ct.Constant):
    pass

def needs_constant_int(x: ct.Constant[int]):
    pass
```

```
_`class`_`cuda.tile.``ConstantAnnotation`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.ConstantAnnotation "Link to this definition")
```

A `typing.Annotated` metadata class indicating that an object shall be [constant embedded](https://docs.nvidia.com/cuda/cutile-python/#execution-constant-embedding).

If an object of this class is passed as a metadata argument to a `typing.Annotated` type hint
on a parameter, then the parameter shall be a constant embedded.

```
`cuda.tile.``Constant`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.Constant "Link to this definition")
```

A type hint indicating that a value shall be [constant embedded](https://docs.nvidia.com/cuda/cutile-python/#execution-constant-embedding).
It can be used either with (`Constant[int]`) or without (`Constant`, meaning a constant of any
type) an underlying type hint.

alias of `Annotated`[`T`, ConstantAnnotation()]
