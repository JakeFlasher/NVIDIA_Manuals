---
title: "Class Member Order"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html#class-member-order"
---

#### [Class Member Order](https://docs.nvidia.com/cutlass/latest/media/docs/cpp#class-member-order)[](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/#class-member-order "Permalink to this headline")

Members within classes and structures should be organized as follows:

1. Type and constant definitions
2. Data members
3. Constructors
4. Other methods

This convention follows the
[CUB library](https://nvlabs.github.io/cub/)
and is also described by
[Howard Hinnant](https://howardhinnant.github.io/classdecl.html).
It also approximates the usual ordering of chapters
in a typical Systems and Controls textbook.
That is, it

1. identifies relevant constants,
2. defines a state-space representation
of the dynamical system under study
(the class’s data members), and then
3. devotes the remaining “chapters” to defining
the system’s dynamical behavior
(the class’s methods).

Here is an example class.

```c++
class A {
public:
  // type definitions
protected:
  // protected type definitions
private:
  // private type definitions

public:
  // data members
protected:
  // protected data members
  // STRONGLY TO BE AVOIDED;
  // please see C++ Core Guidelines
private:
  // private data members

public:
  // methods
protected:
  // protected methods
private:
  // private methods
};
```
