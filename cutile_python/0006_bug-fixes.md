---
title: "Bug Fixes"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html#bug-fixes"
---

### [Bug Fixes](https://docs.nvidia.com/cuda/cutile-python/generated#bug-fixes)[](https://docs.nvidia.com/cuda/cutile-python/generated/#bug-fixes "Permalink to this headline")

- Fix a bug where `nan != nan` returns False.
- Fix “potentially undefined variable `$retval`” error when a helper function
returns after a `while` loop that contains no early return.
- Fix the missing column indicator in error messages when the underlined text is only one
character wide.
- Add a missing check for unpacking a tuple with too many values. For example, `a, b = 1, 2, 3`
now raises an error instead of silently discarding the extra value.
- Fix a bug where the promoted dtype of uint16 and uint64 was incorrectly set to uint32.
