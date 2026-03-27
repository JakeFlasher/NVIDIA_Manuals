---
title: "2.3.4.2. Asynchronous Error Handling"
section: "2.3.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-error-handling"
---

### [2.3.4.2. Asynchronous Error Handling](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#asynchronous-error-handling)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#asynchronous-error-handling "Permalink to this headline")

In a cuda stream, errors may originate from any operation in the stream, including for kernel launches and memory transfers. These errors may not be propagated back to the user at run-time until the stream is synchronized, for example, by waiting for an event or calling `cudaStreamSynchronize()`. There are two ways to find out about errors which may have occurred in a stream.

- Using the function `cudaGetLastError()`  - this function returns and clears the last error encountered in any stream in the current context. An immediate second call to cudaGetLastError() would return `cudaSuccess` if no other error occurred between the two calls.
- Using the function `cudaPeekAtLastError()` - this function returns the last error in the current context, but does not clear it.

Both of these functions return the error as a value of type `cudaError_t`. Printable names names of the errors can be generated using the functions *cudaGetErrorName()* and *cudaGetErrorString()*.

An example of using these functions is shown below:

*Listing 1 Example of using cudaGetLastError() and cudaPeekAtLastError()[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#id6 "Link to this code")*

```c
// Some work occurs in streams.
cudaStreamSynchronize(stream);

// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

> **Tip**
>
> When an error appears at a synchronization, especially in a stream with many operations, it is often difficult to pinpoint exactly where in the stream the error may have occurred. To debug such a situation a useful trick may be to set the environment variable  `CUDA_LAUNCH_BLOCKING=1` and then run the application. The effect of this environment variable is to synchronize after every single kernel launch. This can aid in tracking down which kernel, or transfer caused the error.
> Synchronization can be expensive; applications may run substantially slower when this environment variable is set.
