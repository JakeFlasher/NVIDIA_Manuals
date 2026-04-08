## Description

FP4-quantized Mamba2 output projection that transforms the normalized SSM output back to hidden dimension. Uses FP4 weights with block-scaled quantization (16 elements per block). Performs matrix multiplication from intermediate_size (2688) to hidden_size (2688). Note: dimensions are padded to 2816 to satisfy 128-multiple constraint for FP4 quantization.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [1, M, half_intermediate] | float4_e2m1fn_x2 |
| hidden_states_scale | [1, M, num_intermediate_blocks] | float8_e4m3fn |
| global_decode_scale_input | scalar | float32 |
| weight | [1, hidden_size, half_intermediate] | float4_e2m1fn_x2 |
| weight_scale | [1, hidden_size, num_intermediate_blocks] | float8_e4m3fn |
| global_decode_scale_weight | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [1, M, hidden_size] | float32 |

| # | hidden_size | intermediate_size | sf_vec_size | num_hidden_blocks | num_intermediate_blocks | half_intermediate | M | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1920 | 1920 | 16 | 120 | 120 | 960 | 1664 | 0.0642 | 0.0008 |
| 2 | 1920 | 1920 | 16 | 120 | 120 | 960 | 512 | 0.0601 | 0.0003 |
| 3 | 1920 | 1920 | 16 | 120 | 120 | 960 | 4352 | 0.0738 | 0.0022 |
| 4 | 1920 | 1920 | 16 | 120 | 120 | 960 | 256 | 0.0572 | 0.0001 |
| 5 | 1920 | 1920 | 16 | 120 | 120 | 960 | 4608 | 0.0743 | 0.0023 |
| 6 | 1920 | 1920 | 16 | 120 | 120 | 960 | 2304 | 0.0642 | 0.0012 |
| 7 | 1920 | 1920 | 16 | 120 | 120 | 960 | 1408 | 0.0606 | 0.0007 |
| 8 | 1920 | 1920 | 16 | 120 | 120 | 960 | 3328 | 0.0688 | 0.0017 |
| 9 | 1920 | 1920 | 16 | 120 | 120 | 960 | 4864 | 0.0755 | 0.0025 |
| 10 | 1920 | 1920 | 16 | 120 | 120 | 960 | 640 | 0.0568 | 0.0003 |
| 11 | 1920 | 1920 | 16 | 120 | 120 | 960 | 384 | 0.0567 | 0.0002 |
| 12 | 1920 | 1920 | 16 | 120 | 120 | 960 | 8192 | 0.0896 | 0.0042 |
| 13 | 1920 | 1920 | 16 | 120 | 120 | 960 | 128 | 0.0573 | 0.0001 |
| 14 | 1920 | 1920 | 16 | 120 | 120 | 960 | 1024 | 0.0580 | 0.0005 |
| 15 | 1920 | 1920 | 16 | 120 | 120 | 960 | 3584 | 0.0704 | 0.0018 |
| 16 | 1920 | 1920 | 16 | 120 | 120 | 960 | 2176 | 0.0650 | 0.0011 |

```python
import torch
# --- inlined nvfp4_reference ---
import math
from enum import StrEnum

import torch

class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K).
    """

    BlockWise1x16 = "BlockWise1x16"

class BlockWiseScalerNVFP4:
    def __init__(self, scaling_type: ScalingType = ScalingType.BlockWise1x16):
        self.scaling_type = scaling_type
        self.sf_vec_size = 16
        self.E2M1FN_MAX = 6.0

        # for clamping to nonzero values
        self.E4M3FN_MIN_POS = 2 ** (-9)

    def quantize(self, tensor: torch.Tensor):
        """
        Quantize a high precision tensor to FP4 E2M1 with 1x16 block-wise scaling.

        Args:
            tensor: Input tensor of shape (b, rows, cols) with dtype >= 16 bits (fp16, bf16, fp32)

        Returns:
            Tuple of (quantized_data, scale_factors, global_decode_scale):
            - quantized_data: float4_e2m1fn_x2 tensor of shape (b, rows, cols//2) (2 FP4 values per uint8 byte)
            - scale_factors: FP8 E4M3FN tensor of shape (b, rows, cols//16) with decode scales
            - global_decode_scale: Scalar tensor for global dequantization scaling
        """
        # Compute global amax for the entire tensor
        global_amax = torch.max(torch.abs(tensor)).to(torch.float32)

        # Create scale factors using the helper method (with global scaling)
        decode_scale, global_decode_scale = self._create_scale_factors(
            tensor, global_amax
        )
        # decode_scale shape: (b, rows, cols//16)

        # Now compute encode scale for quantization
        encode_scale = torch.clamp(
            torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
            max=torch.tensor(
                torch.finfo(torch.float32).max,
                device=tensor.device,
                dtype=torch.float32,
            ),
        )
        # encode_scale shape: (b, rows, cols//16)

        # Get tensor dimensions
        b, rows, cols = tensor.shape

        # Reshape to blocks:
        # Scale and quantize
        scaled_x = tensor.view(b, rows, cols // self.sf_vec_size, self.sf_vec_size).to(
            torch.float32
        ) * encode_scale.unsqueeze(-1)
        clipped_x = torch.clamp(scaled_x, -self.E2M1FN_MAX, self.E2M1FN_MAX)

        # Cast to FP4 (packed format): (b, rows, cols//2)
        quantized_data = self._cast_to_fp4x2(clipped_x.reshape(b, rows, cols))

        return quantized_data, decode_scale, global_decode_scale

    def _cast_to_fp4x2(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte).

        Args:
            x: Input tensor of shape (b, rows, cols) with values in range [-6, 6]

        Returns:
            uint8 tensor of shape (b, rows, cols//2) with packed FP4 values
        """
        # FP4 E2M1 encoding
        result = torch.zeros_like(x, dtype=torch.uint8)

        # Positive values
        result[(x >= 0.0) & (x <= 0.25)] = 0
        result[(x > 0.25) & (x < 0.75)] = 1
        result[(x >= 0.75) & (x <= 1.25)] = 2
        result[(x > 1.25) & (x < 1.75)] = 3
        result[(x >= 1.75) & (x <= 2.5)] = 4
        result[(x > 2.5) & (x < 3.5)] = 5
        result[(x >= 3.5) & (x <= 5.0)] = 6
        result[x > 5.0] = 7

        # Negative values
        result[(x >= -0.25) & (x < 0.0)] = 8
        result[(x < -0.25) & (x > -0.75)] = 9
        result[(x <= -0.75) & (x >= -1.25)] = 10
        result[(x < -1.25) & (x > -1.75)] = 11
        result[(x <= -1.75) & (x >= -2.5)] = 12
        result[(x < -2.5) & (x > -3.5)] = 13
        result[(x <= -3.5) & (x >= -5.0)] = 14
        result[x < -5.0] = 15

        # Pack two FP4 values into one byte along cols dimension (dim 1)
        # Input shape: (b, rows, colls)
        # first value (even cols) in low nibble, second value (odd cols) in high nibble
        packed = result[..., ::2] + result[..., 1::2] * 16
        return packed.view(torch.float4_e2m1fn_x2)

    def _create_scale_factors(self, tensor: torch.Tensor, global_amax: torch.Tensor):
        """
        Create scale factors for a given tensor with optional global scaling.

        Args:
            tensor: Input tensor of shape (b, rows, cols)
            global_amax: Optional global max absolute value for global scaling

        Returns:
            Tuple of (decode_scales, global_decode_scale):
            - decode_scales: FP8 E4M3FN tensor of shape (b, rows, cols // 16)
            - global_decode_scale: Scalar tensor (1.0 if no global scaling applied)
        """
        if tensor.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {tensor.dim()}")
        b, rows, cols = tensor.shape

        if cols % self.sf_vec_size != 0:
            raise ValueError(
                f"Cols must be a multiple of {self.sf_vec_size}, got {cols}"
            )

        # Compute max over the block dimensions
        blocked = tensor.view(b, rows, cols // self.sf_vec_size, self.sf_vec_size)
        block_max = blocked.abs().amax(dim=-1)  # (b, rows, cols//16)

        # Compute decode scale: block_max / E2M1_MAX
        decode_scale = block_max / self.E2M1FN_MAX
        decode_scale = decode_scale.to(torch.float32)

        # Apply global scaling (matches nvfp4_tekit.py logic)
        FLOAT8_E4M3_MAX = torch.tensor(448.0, device=tensor.device, dtype=torch.float32)

        # Global encode scale to fit decode scales into FP8 range
        global_encode_scale = torch.div(FLOAT8_E4M3_MAX * self.E2M1FN_MAX, global_amax)
        global_encode_scale = torch.clamp(
            global_encode_scale,
            max=torch.tensor(
                torch.finfo(torch.float32).max,
                device=tensor.device,
                dtype=torch.float32,
            ),
        )

        # Avoid division by zero
        if global_encode_scale == 0.0:
            global_encode_scale = torch.tensor(
                1.0, device=tensor.device, dtype=torch.float32
            )

        global_decode_scale = torch.div(1.0, global_encode_scale)

        # Scale decode_scale to fit in FP8 E4M3FN range
        decode_scale = decode_scale * global_encode_scale
        decode_scale = torch.clamp(
            decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX
        )

        # Convert to FP8 E4M3FN - keep batch-first format (b, rows, cols//16)
        decode_scale = decode_scale.to(dtype=torch.float8_e4m3fn)

        return decode_scale, global_decode_scale

    def convert_to_blocked_format_for_pytorch_scaled_mm(
        self, scale_factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert scale factor tensor to the "blocked" layout expected by torch._scaled_mm.

        Args:
            scale_factors: FP8_E4M3FN Tensor of shape (k//16, m_or_n, b) with strides ((k//16)*m_or_n, m_or_n, 1).

        NOTE: This assumes `rows` is a multiple of 128 and `cols` is a multiple of 4.
        """
        if scale_factors.dim() != 2:
            raise ValueError(
                f"Scale factors must have 2 dimensions, got {scale_factors.dim()}"
            )
        rows, cols = scale_factors.shape
        if rows % 128 != 0:
            raise ValueError(f"Rows must be a multiple of 128, got {rows}")
        if cols % 4 != 0:
            raise ValueError(f"Cols must be a multiple of 4, got {cols}")
        n_row_blocks = math.ceil(rows / 128)
        n_col_blocks = math.ceil(cols / 4)

        blocks = scale_factors.view(n_row_blocks, 128, n_col_blocks, 4).permute(
            0, 2, 1, 3
        )
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        return rearranged.flatten()

class CuBLASRefBlockwiseGemm:
    def __init__(self, b: int, m: int, n: int, k: int):
        self.b = b
        self.m = m
        self.n = n
        self.k = k
        self.sf_vec_size = 16
        self.sf_k = k // self.sf_vec_size
        self.scaler = BlockWiseScalerNVFP4()

        if self.k % self.sf_vec_size != 0:
            raise ValueError(
                f"K must be a multiple of {self.sf_vec_size}, got {self.k}"
            )
        if self.m % 128 != 0:
            raise ValueError(f"M must be a multiple of 128, got {self.m}")
        if self.n % 128 != 0:
            raise ValueError(f"N must be a multiple of 128, got {self.n}")

    def _validate_shapes(self, mat_a: torch.Tensor, mat_b: torch.Tensor):
        if mat_a.dim() != 3:
            raise ValueError(
                f"mat_a must have 3 dimensions (batch, m, k), got {mat_a.dim()}"
            )
        if mat_b.dim() != 3:
            raise ValueError(
                f"mat_b must have 3 dimensions (batch, n, k), got {mat_b.dim()}"
            )

        # the e2m1 tensor has 2 fp4 values per byte, so the shape is (b, m, k//2)
        if mat_a.shape != (self.b, self.m, self.k // 2):
            raise ValueError(
                f"mat_a must have shape (B, M, K//2) = ({self.b}, {self.m}, {self.k // 2}), got {mat_a.shape}"
            )
        if mat_b.shape != (self.b, self.n, self.k // 2):
            raise ValueError(
                f"mat_b must have shape (B, N, K//2) = ({self.b}, {self.n}, {self.k // 2}), got {mat_b.shape}"
            )

    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        global_decode_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        global_decode_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ):
        """
        Perform scaled matrix multiplication: C = (A * scale_a * global_decode_a) @ (B * scale_b * global_decode_b)

        Args:
            mat_a: Quantized matrix A of shape (b, m, k//2)
            mat_b: Quantized matrix B of shape (b, n, k//2)
            scale_a: FP8 E4M3FN scale factors for A of shape (b, m, k//16)
            scale_recipe_a: Scaling type for A (must be BlockWise1x16)
            scale_b: FP8 E4M3FN scale factors for B of shape (b, n, k//16)
            scale_recipe_b: Scaling type for B (must be BlockWise1x16)
            global_decode_a: Optional global decode scale for A (scalar tensor)
            global_decode_b: Optional global decode scale for B (scalar tensor)
            bias: Optional bias tensor (not supported yet)
            output_dtype: Output data type (default: float32)

        Returns:
            Output tensor of shape (b, m, n)
        """
        self._validate_shapes(mat_a, mat_b)

        if (
            scale_recipe_a != ScalingType.BlockWise1x16
            or scale_recipe_b != ScalingType.BlockWise1x16
        ):
            raise ValueError(
                f"Only BlockWise1x16 is supported for scale_recipe, but got {scale_recipe_a} and {scale_recipe_b}"
            )
        if bias is not None:
            raise ValueError("Bias is not supported yet.")

        # Compute combined global scale: global_decode_a * global_decode_b
        global_scale = global_decode_a * global_decode_b

        out = torch.empty(
            (self.m, self.n, self.b), device=mat_a.device, dtype=output_dtype
        )

        mat_a = mat_a.permute(1, 2, 0)
        mat_b = mat_b.permute(1, 2, 0)
        scale_a = scale_a.permute(1, 2, 0)
        scale_b = scale_b.permute(1, 2, 0)

        for b in range(self.b):
            # Extract scales for current batch: (m, k//16, b) -> (m, k//16)
            scale_a_blocked = (
                self.scaler.convert_to_blocked_format_for_pytorch_scaled_mm(
                    scale_a[:, :, b]
                )
            )
            scale_b_blocked = (
                self.scaler.convert_to_blocked_format_for_pytorch_scaled_mm(
                    scale_b[:, :, b]
                )
            )

            # Perform scaled matmul with block-wise scales
            result = torch._scaled_mm(
                mat_a[:, :, b],
                mat_b[:, :, b].transpose(0, 1),  # (n, k) -> (k, n) without moving data
                scale_a_blocked,
                scale_b_blocked,
                bias=None,
                out_dtype=output_dtype,
            )

            # Apply global decode scale: result * global_decode_a * global_decode_b
            out[:, :, b] = result * global_scale

        return out.permute(2, 0, 1)

# --- end inlined nvfp4_reference ---

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate input tensors for NVFP4 Mamba2 output projection."""
    M = axes_and_scalars["M"]
    hidden_size = 1920
    intermediate_size = 1920
    
    # Create random FP32 tensors and quantize them
    scaler = BlockWiseScalerNVFP4()
    
    # Input hidden states: [1, M, intermediate_size]
    hidden_fp32 = torch.randn(1, M, intermediate_size, dtype=torch.float32, device=device)
    hidden_quantized, hidden_scale, global_decode_input = scaler.quantize(hidden_fp32)
    
    # Weight: [1, hidden_size, intermediate_size]
    weight_fp32 = torch.randn(1, hidden_size, intermediate_size, dtype=torch.float32, device=device)
    weight_quantized, weight_scale, global_decode_weight = scaler.quantize(weight_fp32)
    
    return {
        "hidden_states": hidden_quantized,
        "hidden_states_scale": hidden_scale,
        "global_decode_scale_input": global_decode_input.item(),
        "weight": weight_quantized,
        "weight_scale": weight_scale,
        "global_decode_scale_weight": global_decode_weight.item(),
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    global_decode_scale_input: float,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_decode_scale_weight: float,
):
    """FP4 block-scaled output projection.
    
    Args:
        hidden_states: [1, M, K//2] FP4 packed input
        hidden_states_scale: [1, M, K//16] FP8 scales
        global_decode_scale_input: global scale for input
        weight: [1, N, K//2] FP4 packed weight
        weight_scale: [1, N, K//16] FP8 scales
        global_decode_scale_weight: global scale for weight
    
    Returns:
        output: [1, M, N] float32 result
    """
    _, M, half_K = hidden_states.shape
    K = half_K * 2  # intermediate_size
    _, N, _ = weight.shape  # hidden_size
    
    device = hidden_states.device
    
    # Convert global scales to tensors
    global_scale_input = torch.tensor(global_decode_scale_input, dtype=torch.float32, device=device)
    global_scale_weight = torch.tensor(global_decode_scale_weight, dtype=torch.float32, device=device)
    
    # Use CuBLASRefBlockwiseGemm for batched FP4 GEMM
    # hidden_states: [1, M, K//2] @ weight.T: [1, K//2, N] -> [1, M, N]
    gemm_ref = CuBLASRefBlockwiseGemm(b=1, m=M, n=N, k=K)
    
    result = gemm_ref.scaled_mm(
        mat_a=hidden_states,
        mat_b=weight,
        scale_a=hidden_states_scale,
        global_decode_a=global_scale_input,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=weight_scale,
        global_decode_b=global_scale_weight,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32,
    )
    
    return result

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"M": 256},
        device=torch.device("cuda:0")
    )
    out = run(**inputs)
    print(out.shape)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000748 ms |
| - | Scoring Baseline | 0.500000 | 0.065214 ms |
| - | Reference Implementation | 0.284380 | 0.163010 ms |
