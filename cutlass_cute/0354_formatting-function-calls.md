---
title: "Formatting function calls"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#formatting-function-calls"
---

#### [Formatting function calls](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#formatting-function-calls)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#formatting-function-calls "Permalink to this headline")

When calling a function or function object with a long name,
break the line right after the invoking open parenthesis.
Here are some examples.

```c++
detail::very_long_function_object_name<TemplateArgument>{}(
  params.long_parameter_name, some_operator.another_long_function_name());

detail::an_even_longer_function_object_name<TemplateArgument1, TemplateArgument2>{}(
  params.long_parameter_name, some_operator.long_member_function_name(),
  another_operator.another_long_member_function_name(x, y, z));
```
