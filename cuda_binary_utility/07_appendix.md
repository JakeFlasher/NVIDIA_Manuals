---
title: "7. Appendix"
section: "7"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#appendix"
---

# [7. Appendix](https://docs.nvidia.com/cuda/cuda-binary-utilities#appendix)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#appendix "Permalink to this headline")

## [7.1. JSON Format](https://docs.nvidia.com/cuda/cuda-binary-utilities#json-format)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#json-format "Permalink to this headline")

The output of `nvdisasm` is human-readable text which is not formatted for machine consumption.
Any tool consuming the output of nvdisasm must parse the human-readable text which can be slow
and any minor changes to the text can break the parser.

JSON-based format provides an efficient and extensible method to output machine readable data from `nvdisasm`.
The option `-json` can be used to produce a JSON document that adheres to the following JSON schema definition.

```text
{
    "$id": "https://nvidia.com/cuda/cuda-binary-utilities/index.html#json-format",
    "description": "A JSON schema for NVIDIA CUDA disassembler. The $id attribute is not a real URL but a unique identifier for the schema",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "A JSON schema for NVIDIA CUDA disassembler",
    "version": "13-1-0",
    "type": "array",
    "minItems": 2,
    "prefixItems": [
        {
            "$ref": "#/$defs/metadata"
        },
        {
            "description": "A list of CUDA functions",
            "type": "array",
            "minItems": 1,
            "items": {
                "$ref": "#/$defs/function"
            }
        }
    ],
    "$defs": {
        "metadata": {
            "type": "object",
            "properties": {
                "ELF": {
                    "$ref": "#/$defs/elf-metadata"
                },
                "SM": {
                    "type": "object",
                    "properties": {
                        "version": {
                            "$ref": "#/$defs/sm-version"
                        }
                    },
                    "required": [
                        "version"
                    ]
                },
                "SchemaVersion": {
                    "$ref": "#/$defs/version"
                },
                "Producer": {
                    "type": "string",
                    "description": "Name and version of the CUDA disassembler tool",
                    "maxLength": 1024
                },
                "Description": {
                    "type": "string",
                    "description": "A description that may be empty",
                    "maxLength": 1024
                },
                ".note.nv.cuinfo": {
                    "$ref": "#/$defs/Elf64_NV_CUinfo"
                },
                ".note.nv.tkinfo": {
                    "$ref": "#/$defs/Elf64_NV_TKinfo"
                }
            },
            "required": [
                "ELF",
                "SM",
                "SchemaVersion",
                "Producer",
                "Description"
            ]
        },
        "elf-metadata": {
            "type": "object",
            "properties": {
                "layout-id": {
                    "description": "Indicates the layout of the ELF file, part of the ELF header flags. Undocumented enum",
                    "type": "integer"
                },
                "ei_osabi": {
                    "description": "Operating system/ABI identification",
                    "type": "integer"
                },
                "ei_abiversion": {
                    "description": "ABI version",
                    "type": "integer"
                }
            },
            "required": [
                "layout-id",
                "ei_osabi",
                "ei_abiversion"
            ]
        },
        "sm-version": {
            "type": "object",
            "properties": {
                "major": {
                    "type": "integer"
                },
                "minor": {
                    "type": "integer"
                }
            },
            "required": [
                "major",
                "minor"
            ]
        },
        "version": {
            "type": "object",
            "properties": {
                "major": {
                    "type": "integer"
                },
                "minor": {
                    "type": "integer"
                },
                "revision": {
                    "type": "integer"
                }
            },
            "required": [
                "major",
                "minor",
                "revision"
            ]
        },
        "sass-instruction-attribute": {
            "type": "object",
            "additionalProperties": {
                "type": "string"
            }
        },
        "sass-instruction": {
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "description": "Instruction predicate"
                },
                "opcode": {
                    "type": "string",
                    "description": "The instruction opcode. May be empty to indicate a gap between non-contiguous instructions"
                },
                "operands": {
                    "type": "string",
                    "description": "Instruction operands separated by commas"
                },
                "extra": {
                    "type": "string",
                    "description": "Optional field"
                },
                "other-attributes": {
                    "type": "object",
                    "description": "Additional instruction attributes encoded as a map of string:string key-value pairs. Example: {'control-flow': 'True'}",
                    "properties": {
                        "control-flow": {
                            "const": ["True"],
                            "description": "True if the instruction is a control flow instruction"
                        },
                        "subroutine-call": {
                            "const": ["True"],
                            "description": "True if the instruction is a subroutine call"
                        }
                    }
                },
                "other-flags": {
                    "type": "array",
                    "description": "Aditional instruction attributes encoded as a list strings",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "opcode"
            ]
        },
        "function": {
            "type": "object",
            "properties": {
                "function-name": {
                    "type": "string"
                },
                "start": {
                    "type": "integer",
                    "description": "The function's start virtual address"
                },
                "length": {
                    "type": "integer",
                    "description": "The function's length in bytes"
                },
                "other-attributes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "sass-instructions": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/sass-instruction"
                    }
                }
            },
            "required": [
                "function-name",
                "start",
                "length",
                "sass-instructions"
            ]
        },
        "Elf64_NV_CUinfo": {
            "type": "object",
            "properties": {
                "nv_note_cuinfo": {
                    "type": "integer"
                },
                "nv_note_cuinfo_virt_sm": {
                    "type": "integer"
                },
                "nv_note_cuinfo_toolVersion": {
                    "type": "integer"
                }
            },
            "required": [
                "nv_note_cuinfo",
                "nv_note_cuinfo_virt_sm",
                "nv_note_cuinfo_toolVersion"
            ]
        },
        "Elf64_NV_TKinfo": {
            "type": "object",
            "properties": {
                "nv_note_tkinfo": {
                    "type": "integer"
                },
                "tki_objFname": {
                    "type": "string"
                },
                "tki_toolName": {
                    "type": "string"
                },
                "tki_toolVersion": {
                    "type": "string"
                },
                "tki_toolBranch": {
                    "type": "string"
                },
                "tki_toolOptions": {
                    "type": "string"
                }
            },
            "required": [
                "nv_note_tkinfo",
                "tki_objFname",
                "tki_toolName",
                "tki_toolVersion",
                "tki_toolBranch",
                "tki_toolOptions"
            ]
        }
    }
}
```

**Notes about sass-instruction objects**

- The `other-attributes` object may contain `"control-flow": "True"` key-pair to indicate control flow instructions and `"subroutine-call": "True"` key-pair to indicate subroutine call instructions.
- The address of the nth (0-based) SASS instruction can be computed as start + n * instruction size . The instruction size is 16 bytes.
- The JSON list may contain empty instruction objects; these objects count towards the instruction index, as they are indicating gaps between non-contiguous instructions.
- An empty instruction object has the single field `opcode` with an empty string value : `"opcode": ""`

Here’s a sample output from `nvdisasm -json`

```text
[
    // First element in the list: Metadata
    {
        // ELF Metadata
        "ELF": {
            "layout-id": 4,
            "ei_osabi": 51,
            "ei_abiversion": 7
        },
        // SASS code SM version: SM89 (16 bytes instructions)
        "SM": {
            "version": {
                "major": 8,
                "minor": 9
            }
        },
        "SchemaVersion": {
            "major": 12,
            "minor": 8,
            "revision": 0
        },
        // Release details of nvdisasm
        "Producer": "nvdisasm V12.8.14 Build r570_00.r12.8/compiler.35033008_0",
        "Description": ""
    },
    // Second element in the list: Functions
    [
        {
            "function-name": "_Z10exampleKernelv",
            // Function start address
            "start": 0,
            // Function length in bytes
            "length": 384,
            "other-attributes": [],
            // SASS instructions
            "sass-instructions": [
                {
                    // Instruction at 0x00
                    "opcode": "IMAD.MOV.U32",
                    "operands": "R1,RZ,RZ,c[0x0][0x28]"
                },
                {
                    // Instruction at 0x10 (16 bytes increment)
                    "opcode": "MOV",
                    "operands": "R0,0x0"
                },
                {
                    // Instruction at 0x20
                    "opcode": "IMAD.MOV.U32",
                    "operands": "R4,RZ,RZ,c[0x4][0x8]"
                },
                // [...]
                {
                    "opcode": "CALL.ABS.NOINC",
                    "operands": "R2",
                    // other-attributes is an optional that can indicate control flow instructions
                    "other-attributes": {
                        "control-flow": "True",
                        "subroutine-call": "True"
                    }
                },
                {
                    "opcode": "EXIT",
                    "other-attributes": {
                        "control-flow": "True"
                    }
                },
                {
                    "opcode": "NOP"
                }
            ]
        }
    ]
]
```
