---
title: "Operating system settings"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html#operating-system-settings"
---

## [Operating system settings](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build#operating-system-settings)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/#operating-system-settings "Permalink to this headline")

By default, Windows restricts the maximum file path length (`MAX_PATH`) to 260 characters.
CUTLASS has many files and directory paths that challenge this requirement.
As a result, CUTLASS is unlikely to build with this default setting.
The choice of source and build directories affect path lengths,
so the kinds of errors and whether they occur may depend on this.
Symptoms may vary, from errors when running `cmake`
(e.g., during the “generating library instances” step) to build failures.

CUTLASS recommends changing the maximum file path length setting
and rebooting the computer before attempting to clone or build CUTLASS.
Windows 10 (as of version 1607) and 11 permit changing this setting
by making sure that the following registry key exists,
and that its value is set to 1.

```console
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled
```

After changing the registry key’s value, reboot the computer first
before attempting to clone or build CUTLASS.

[This Microsoft help article](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)
explains different ways to change the registry setting.
