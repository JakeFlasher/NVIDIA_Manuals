---
title: "Deprecation Process"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/deprecation.html#deprecation-process"
---

## [Deprecation Process](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL#deprecation-process)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/#deprecation-process "Permalink to this headline")

**Step 1 — Soft Deprecation**

When a feature is considered for removal, it is first annotated with the
`@deprecated` decorator or `DeprecationWarning` and documented with a
suggested alternative. At this stage, the feature continues to work normally.

Users are encouraged to provide feedback and describe their use cases.
If there is strong justification, we may keep or redesign the feature.

**Step 2 — Removal (the subsequent release)**

If no valid use cases remain, the deprecated feature will be removed in the
following **minor** release.

> **Note**
>
> The release version follows the format `<major>.<minor>.<patch>`.
