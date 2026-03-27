---
title: "2.3.9.2. Introduction to CUDA Graphs with Stream Capture"
section: "2.3.9.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#introduction-to-cuda-graphs-with-stream-capture"
---

### [2.3.9.2. Introduction to CUDA Graphs with Stream Capture](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#introduction-to-cuda-graphs-with-stream-capture)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#introduction-to-cuda-graphs-with-stream-capture "Permalink to this headline")

CUDA streams allow programs to specify a sequence of operations, kernels or memory copies, in order. Using multiple streams and cross-stream dependencies with `cudaStreamWaitEvent`, an application can specify a full directed acyclic graph (DAG) of operations. Some applications may have a sequence or DAG of operations that needs to be run many times throughout execution.

For this situation, CUDA provides a feature known as CUDA graphs. This section introduces CUDA graphs and one mechanism of creating them called _stream capture_. A more detailed discussion of CUDA graphs is presented in [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs). Capturing or creating a graph can help reduce latency and CPU overhead of repeatedly invoking the same chain of API calls from the host thread. Instead, the APIs to specify the graph operations can be called once, and then the resulting graph executed many times.

CUDA Graphs work in the following way:

1. The graph is _captured_ by the application. This step is done once the first time the graph is executed. The graph can also be manually composed using the CUDA graph API.
2. The graph is _instantiated_. This step is done one time, after the graph is captured. This step can set up all the various runtime structures needed to execute the graph, in order to make launching its components as fast as possible.
3. In the remaining steps, the pre-instantiated graph is executed as many times as required. Since all the runtime structures needed to execute the graph operations are already in place, the CPU overheads of the graph execution are minimized.

*Listing 2 The stages of capturing, instantiating and executing a simple linear graph using CUDA Graphs (from [CUDA Developer Technical Blog](https://developer.nvidia.com/blog/cuda-graphs/), A. Gray, 2019)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#id7 "Link to this code")*

```c
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }

        // End the capture
        cudaStreamEndCapture(stream, &graph);

        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```

Much more detail on CUDA graph is provided in [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs).
