---
title: "VSCode Setup"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html#vscode-setup"
---

## [VSCode Setup](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#vscode-setup)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#vscode-setup "Permalink to this headline")

1. Install the [Official C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
2. Open settings…
  1. `Ctrl+Shift+P` to open the command palette
  2. Enter “C/C++” to filter results
  3. Select “C/C++ Edit Configurations (UI)” (or “… (JSON)” if you feel like editing the raw JSON)
  4. View the documentation for these settings
[here](https://code.visualstudio.com/docs/cpp/customize-cpp-settings)
3. Edit “Include Path” to set up **include paths**. For CUTLASS, this includes the following:
  - `${workspaceFolder}/include`
  - `${workspaceFolder}/tools/util/include`
  - `${workspaceFolder}/examples/common`
  - …others, depending on which files you edit
4. Edit C++ standard to be `c++17`, `gnu++17`, or equivalent.
5. Edit `defines` to define preprocessor variables. See
[Global Config below](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#global-config) for examples. The important
ones include `__CUDACC_VER_MAJOR__`, `__CUDA_ARCH__`, `__CUDA_ARCH_FEAT_SM90_ALL__`. But configure
them according to your target architecture.
6. …and possible edit any other fields for your specific setup.
