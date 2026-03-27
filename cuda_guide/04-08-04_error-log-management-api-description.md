---
title: "4.8.4. API Description"
section: "4.8.4"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management--api-description"
---

## [4.8.4. API Description](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#api-description)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#api-description "Permalink to this headline")

The CUDA Driver provides APIs in two categories for interacting with the Error Log Management feature.

This feature allows developers to register callback functions to be used whenever an error log is generated, where the callback signature is:

```c++
void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)
```

Callbacks are registered with this API:

```c++
CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)
```

Where _userData_ is passed to the callback function without modifications. _callback_out_ should be stored by the caller for use in _cuLogsUnregisterCallback_.

```c++
CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)
```

The other set of API functions are for managing the output of logs. An important concept is the log iterator, which points to the current end of the buffer:

```c++
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)
```

The iterator position can be kept by the calling software in situations where a dump of the entire log buffer is not desired. Currently, the flags parameter must be 0, with additional options reserved for future CUDA releases.

At any time, the error log buffer can be dumped to either a file or memory with these functions:

```c++
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)
```

If _iterator_ is NULL, the entire buffer will be dumped, up to the maximum of 100 entries. If _iterator_ is not NULL, logs will be dumped starting from that entry and the value of _iterator_ will be updated to the current end of the logs, as if _cuLogsCurrent_ had been called. If there have been more than 100 log entries into the buffer, a note will be added at the start of the dump noting this.

The flags parameter must be 0, with additional options reserved for future CUDA releases.

The _cuLogsDumpToMemory_ function has additional considerations:

1. The buffer itself will be null-terminated, but each individual log entry will only be separated by a newline (n) character.
2. The maximum size of the buffer is 25600 bytes.
3. If the value provided in _size_ is not sufficient to store all desired logs, a note will be added as the first entry and the oldest entries that do not fit will not be dumped.
4. After returning, _size_ will contain the actual number of bytes written to the provided buffer.
