---
title: "Enabling Logging"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html#enabling-logging"
---

### [Enabling Logging](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#enabling-logging)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#enabling-logging "Permalink to this headline")

CuTe DSL provides environment variables to control logging level:

```bash
# Enable console logging (default: False)
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False)
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
export CUTE_DSL_LOG_LEVEL=20
```
