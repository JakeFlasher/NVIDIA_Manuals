---
title: "Rounding Modes"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/data.html#rounding-modes"
---

## [Rounding Modes](https://docs.nvidia.com/cuda/cutile-python#rounding-modes)[](https://docs.nvidia.com/cuda/cutile-python/#rounding-modes "Permalink to this headline")

```
_`class`_`cuda.tile.``RoundingMode`[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode "Link to this definition")
```

Rounding mode for floating-point operations.

```
`RN`_`=` `'nearest_even'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.RN "Link to this definition")
```

Rounds the nearest (ties to even).

```
`RZ`_`=` `'zero'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.RZ "Link to this definition")
```

Round towards zero (truncate).

```
`RM`_`=` `'negative_inf'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.RM "Link to this definition")
```

Round towards negative infinity.

```
`RP`_`=` `'positive_inf'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.RP "Link to this definition")
```

Round towards positive infinity.

```
`FULL`_`=` `'full'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.FULL "Link to this definition")
```

Full precision rounding mode.

```
`APPROX`_`=` `'approx'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.APPROX "Link to this definition")
```

Approximate rounding mode.

```
`RZI`_`=` `'nearest_int_to_zero'`_[](https://docs.nvidia.com/cuda/cutile-python/#cuda.tile.RoundingMode.RZI "Link to this definition")
```

Round towards zero to the nearest integer.
