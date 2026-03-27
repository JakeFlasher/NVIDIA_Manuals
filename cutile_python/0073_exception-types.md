---
title: "Exception Types"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/debugging.html#exception-types"
---

## [Exception Types](https://docs.nvidia.com/cuda/cutile-python#exception-types)[](https://docs.nvidia.com/cuda/cutile-python/#exception-types "Permalink to this headline")

```
_`class`_`cuda.tile.``TileSyntaxError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileSyntaxError "Link to this definition")
```

Exception when a python syntax not supported by cuTile is encountered.

```
_`class`_`cuda.tile.``TileTypeError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileTypeError "Link to this definition")
```

Exception when an unexpected type or [data type](https://docs.nvidia.com/cuda/cutile-python/data.html#data-data-types) is encountered.

```
_`class`_`cuda.tile.``TileValueError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileValueError "Link to this definition")
```

Exception when an unexpected python value is encountered.

```
_`class`_`cuda.tile.``TileUnsupportedFeatureError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileUnsupportedFeatureError "Link to this definition")
```

Exception when a feature is not supported by the underlying compiler or
the GPU architecture.

```
_`class`_`cuda.tile.``TileCompilerExecutionError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileCompilerExecutionError "Link to this definition")
```

Exception when `tileiras` compiler throws an error.

```
_`class`_`cuda.tile.``TileCompilerTimeoutError`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileCompilerTimeoutError "Link to this definition")
```

Exception when `tileiras` compiler timeout limit is exceeded.
