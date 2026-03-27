---
title: "CTK 13.2 features"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html#ctk-13-2-features"
---

### [CTK 13.2 features](https://docs.nvidia.com/cuda/cutile-python/generated#ctk-13-2-features)[](https://docs.nvidia.com/cuda/cutile-python/generated/#ctk-13-2-features "Permalink to this headline")

- Support Ampere and Ada (sm80 family) GPUs.
- Support `pip install cuda-tile[tileiras]` to use `tileiras` from Python environment
without system-wide CTK installation.
- Add `ct.atan2(y, x)` operation for computing the arctangent of y/x.
- Add optional `rounding_mode` parameter for `ct.tanh()`, supporting `RoundingMode.FULL` and
`RoundingMode.APPROX`.
- Compiling FP8 operations for sm80 family GPUs will raise `TileUnsupportedFeatureError`.
- Setting `opt_level=0` on `ct.kernel` is no longer required for `ct.printf()` and `ct.print()`.
