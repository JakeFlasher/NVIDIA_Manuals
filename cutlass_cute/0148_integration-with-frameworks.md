---
title: "Integration with Frameworks"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#integration-with-frameworks"
---

# [Integration with Frameworks](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#integration-with-frameworks)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#integration-with-frameworks "Permalink to this headline")

In order to facilitate the integration of CUTLASS Python with popular frameworks, we leverage the
[DLPack protocol](https://github.com/dmlc/dlpack) and transform tensors originating from these
frameworks to CuTe tensors. The present page documents the conventions, the API available to the
user, and provide example code snippets for common usage patterns. We also provide a section on how to
bypass the DLPack protocol and directly call the JIT function.
