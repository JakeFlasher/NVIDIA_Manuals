---
title: "C++ Integration with Dynamic Loading"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html#c-integration-with-dynamic-loading"
---

### [C++ Integration with Dynamic Loading](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#c-integration-with-dynamic-loading)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#c-integration-with-dynamic-loading "Permalink to this headline")

Dynamically load pre-compiled object files or shared libraries at runtime. By including the `CuteDSLRuntime.h` header, you can load the module, look up exported functions, and invoke them.

```cpp
#include "CuteDSLRuntime.h"
#include <cuda_runtime.h>

void run_print_tensor() {
    // Load module from shared library
    CuteDSLRT_Module_t *module = nullptr;
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/libprint_tensor_example.so"
    );
    // or
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/print_tensor_example.o"
    );
    check_error(err);

    // Lookup function
    CuteDSLRT_Function_t *func = nullptr;
    err = CuteDSLRT_Module_Get_Function(&func, module, "print_tensor");
    check_error(err);

    // Prepare arguments, matching the argument type defined in the header file
    typedef struct {
        void *data;
        int32_t dynamic_shapes[2];
        int64_t dynamic_strides[1];
    } print_tensor_Tensor_a_t;

    print_tensor_Tensor_a_t tensor_a;
    tensor_a.data = nullptr;
    tensor_a.dynamic_shapes[0] = 32;
    tensor_a.dynamic_shapes[1] = 16;
    tensor_a.dynamic_strides[0] = 16;

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Call the function; the runtime function accepts packed arguments, refer to the wrapper in the header file
    int ret;
    void* args[] = {&tensor_a, &stream, &ret};
    err = CuteDSLRT_Function_Run(func, args, 3);
    check_error(err);
    cudaStreamSynchronize(stream);

    // Cleanup
    CuteDSLRT_Module_Destroy(module);
    cudaStreamDestroy(stream);
}
```

The `CuteDSLRuntime.h` header file can be found in `<wheel_install_path>/include`. It includes:

- The `CuteDSLRT_Error_t` type: Indicates error status.
- The `CuteDSLRT_Module_Load` function: Loads the module.
- The `CuteDSLRT_Module_Get_Function` function: Gets a function from the loaded module. The runtime API will load the CUDA module for kernel execution.
- The `CuteDSLRT_Function_Run` function: Runs the function.
- The `CuteDSLRT_Module_Destroy` function: Destroys the module.

The compilation of the C++ executable requires the `libcute_dsl_runtime.so` library which is involved in `<wheel_install_path>/lib`, along with the CUDA driver and runtime libraries, to function properly.
