---
title: "Overview"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html#ide_setup--overview"
---

## [Overview](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#overview)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#overview "Permalink to this headline")

In order for any intellisense tool to work with CUTLASS, the following things need to be configured with it:

- Include paths, i.e. where the compiler (or in this case, the intellisense tool) should look for header files
- Compiler flags; especially the C++ standard (`--std`)
- Preprocessor variables; especially CUDA-related ones

One usually needs to configure the above variables in a settings file. Below, two config approaches are described:
for VSCode, and for any editor that uses the clangd language server, which includes
Vim, Emacs, NeoVim, Sublime Text, and so on. Note that VSCode can also be configured to use clangd.
It might be worth setting up clangd for VSCode rather than the default intellisense,
and you might see faster responses and more stable performance with clangd.
