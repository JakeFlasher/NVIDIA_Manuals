---
title: "Output"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#output"
---

## [Output](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#output)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#output "Permalink to this headline")

By default, runtime and computed GFLOP/s are reported for each operation and problem size. Additionally,
a table of comma separated values are reported at the end of the execution. This may be output to a file
with the `--output=<filename.csv>` command line option as shown:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv
```

To faclitate generation of pivot tables and charts, additional columns may be prepended with the
`--tags=<column>:<value>` option. One or more tags may be specified using a comma-delimited list.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv \
                                    --tags=cutlass:2.2,date:2020-06-08
```
