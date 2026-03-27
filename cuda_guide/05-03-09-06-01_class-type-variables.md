---
title: "5.3.9.6.1. Class-type Variables"
section: "5.3.9.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#class-type-variables"
---

#### [5.3.9.6.1. Class-type Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#class-type-variables)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#class-type-variables "Permalink to this headline")

A variable definition with `__device__`, `__constant__`, `__managed__` or `__shared__`  memory space cannot have a class type with a non-empty constructor or a non-empty destructor. A constructor for a class type is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

- The constructor function has been defined.
- The constructor function has no parameters, an empty initializer list, and an empty compound statement function body.
- Its class has no `virtual` functions, `virtual` base classes, or non-`static` data member initializers.
- The default constructors of all of its base classes can be considered empty.
- For all non-`static` data members of the class that are of a class type (or an array thereof), the default constructors can be considered empty.

A class’s destructor is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

- The destructor function has been defined.
- The destructor function body is an empty compound statement.
- Its class has no `virtual` functions or `virtual` base classes.
- The destructors of all of its base classes can be considered empty.
- For all non-`static` data members of the class that are of a class type (or an array thereof), the destructor can be considered empty.
