---
title: "clangd Setup"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html#clangd-setup"
---

## [clangd Setup](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#clangd-setup)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#clangd-setup "Permalink to this headline")

`clangd` is a C++ language server that is part of the LLVM project. You must first set it up your specific IDE:

- `clangd` official [documentation](https://clangd.llvm.org/installation#editor-plugins) for editor setup.
- NeoVim setup is possible through [lsp](https://neovim.io/doc/user/lsp.html) and either manually installing clangd or
using an installation manager like Mason.

Then, one needs to edit the config ([documentation](https://clangd.llvm.org/config)). One typically has a
**global** and a **per-project** config.
