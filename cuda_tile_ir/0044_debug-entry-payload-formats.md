---
title: "Debug Entry Payload Formats"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#debug-entry-payload-formats"
---

#### [Debug Entry Payload Formats](https://docs.nvidia.com/cuda/tile-ir/latest/sections#debug-entry-payload-formats)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#debug-entry-payload-formats "Permalink to this headline")

Each debug entry in `diData` begins with a `debugEntryType` byte followed by type-specific data:

```text
debugEntry {
  debugEntryType : byte     // Identifies the debug info type
  entryData : byte[]        // Type-specific payload (format below)
}
```

The following debug entry types are supported:

**Unknown Debug Info** (`debugEntryType` = 0x00)

```text
debugEntryType : varint = 0x00  // Unknown debug info
// No additional payload
```

**DICompileUnit** (`debugEntryType` = 0x01)

```text
debugEntryType : varint = 0x01  // DICompileUnit
language : varint               // Source language identifier
fileIndex : varint              // Index of associated DIFile
producer : varint               // String index for compiler producer
optimized : byte                // 0x00=false, 0x01=true
emissionKind : varint           // Emission kind enumeration
```

**DIFile** (`debugEntryType` = 0x02)

```text
debugEntryType : varint = 0x02  // DIFile
filename : varint               // String index for filename
directory : varint              // String index for directory
```

**DILexicalBlock** (`debugEntryType` = 0x03)

```text
debugEntryType : varint = 0x03  // DILexicalBlock
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of parent scope (DIFile or DISubprogram)
```

**DILoc** (`debugEntryType` = 0x04)

```text
debugEntryType : varint = 0x04  // DILoc (source location)
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of scope (DISubprogram, DILexicalBlock, etc.)
inlinedAtIndex : varint         // Index of inlined location (0 if not inlined)
```

**DISubprogram** (`debugEntryType` = 0x05)

```text
debugEntryType : varint = 0x05  // DISubprogram (function debug info)
name : varint                   // String index for function name
linkageName : varint            // String index for linkage name
fileIndex : varint              // Index of associated DIFile
line : varint                   // Line number where function is defined
typeIndex : varint              // Index of function type
scopeLineIndex : varint         // Line number where scope begins
flags : varint                  // Function flags (visibility, etc.)
unitIndex : varint              // Index of associated DICompileUnit
```

**CallSite** (`debugEntryType` = 0x06)

```text
debugEntryType : varint = 0x06  // CallSite location
calleeIndex : varint            // Index of called location
callerIndex : varint            // Index of calling location
```
