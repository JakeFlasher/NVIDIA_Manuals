---
title: "Environment Variables"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/debugging.html#environment-variables"
---

## [Environment Variables](https://docs.nvidia.com/cuda/cutile-python#environment-variables)[](https://docs.nvidia.com/cuda/cutile-python/#environment-variables "Permalink to this headline")

The following environment variables are useful when
the above exceptions are encountered during kernel
development.

Set `CUDA_TILE_ENABLE_CRASH_DUMP=1` to enable dumping
an archive including the TileIR bytecode
for submitting a bug report on [`TileCompilerExecutionError`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileCompilerExecutionError "cuda.tile.TileCompilerExecutionError")
or [`TileCompilerTimeoutError`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileCompilerTimeoutError "cuda.tile.TileCompilerTimeoutError").

Set `CUDA_TILE_COMPILER_TIMEOUT_SEC` to limit the
time the TileIR compiler *tileiras* can take.

Set `CUDA_TILE_LOGS=CUTILEIR` to print cuTile Python
IR during compilation to stderr. This is useful when
debugging [`TileTypeError`](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.TileTypeError "cuda.tile.TileTypeError").

Set `CUDA_TILE_TEMP_DIR` to configure the directory
for storing temporary files.

Set `CUDA_TILE_CACHE_DIR` to configure the directory
for the bytecode-to-cubin disk cache. Compiled cubins
are cached here to avoid recompilation of unchanged
kernels. Set to `0`, `off`, `none`, or an empty
string to disable caching. Defaults to
`~/.cache/cutile-python`.

Set `CUDA_TILE_CACHE_SIZE` to configure the maximum
disk cache size in bytes. Oldest entries are evicted
when the cache exceeds this limit. Defaults to
2 GB (2147483648).
