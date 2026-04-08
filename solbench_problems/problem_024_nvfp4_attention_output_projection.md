## Description

FP4-quantized attention output projection (o_proj) that transforms concatenated multi-head attention outputs back to hidden dimension. Uses FP4 block-scaled GEMM with 16 elements per FP8 scale factor. Computes C = (A * scale_a * global_a) @ (B^T * scale_b * global_b) where A is the input activation and B is the weight matrix.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [M, input_dim] | float32 |
| weight_fp4 | [hidden_size_padded, 2048] | float4_e2m1fn_x2 |
| weight_scale_fp8 | [hidden_size_padded, sf_k] | float8_e4m3fn |
| weight_global_decode_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [M, hidden_size_padded] | float32 |

| # | hidden_size | input_dim | sf_vec_size | hidden_size_padded | sf_k | B | L | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2688 | 4096 | 16 | 2816 | 256 | 64 | 256 | 0.8405 | 0.0185 |
| 2 | 2688 | 4096 | 16 | 2816 | 256 | 8 | 2048 | 0.7550 | 0.0062 |
| 3 | 2688 | 4096 | 16 | 2816 | 256 | 1 | 2048 | 0.2356 | 0.0031 |
| 4 | 2688 | 4096 | 16 | 2816 | 256 | 16 | 128 | 0.2254 | 0.0492 |
| 5 | 2688 | 4096 | 16 | 2816 | 256 | 8 | 1024 | 0.4508 | 0.0246 |
| 6 | 2688 | 4096 | 16 | 2816 | 256 | 1 | 8192 | 0.4548 | 0.0062 |
| 7 | 2688 | 4096 | 16 | 2816 | 256 | 1 | 256 | 0.2138 | 0.0985 |
| 8 | 2688 | 4096 | 16 | 2816 | 256 | 2 | 128 | 0.2479 | 0.0123 |
| 9 | 2688 | 4096 | 16 | 2816 | 256 | 16 | 256 | 0.3012 | 0.0062 |
| 10 | 2688 | 4096 | 16 | 2816 | 256 | 8 | 256 | 0.2273 | 0.0123 |
| 11 | 2688 | 4096 | 16 | 2816 | 256 | 4 | 2048 | 0.4544 | 0.0031 |
| 12 | 2688 | 4096 | 16 | 2816 | 256 | 2 | 512 | 0.2122 | 0.0369 |
| 13 | 2688 | 4096 | 16 | 2816 | 256 | 32 | 256 | 0.4541 | 0.0185 |
| 14 | 2688 | 4096 | 16 | 2816 | 256 | 1 | 128 | 0.2139 | 0.0092 |
| 15 | 2688 | 4096 | 16 | 2816 | 256 | 1 | 1024 | 0.2219 | 0.0492 |
| 16 | 2688 | 4096 | 16 | 2816 | 256 | 64 | 512 | 1.3558 | 0.0015 |

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
    """Generate inputs for FP4 attention output projection."""
    B = axes_and_scalars["B"]
    L = axes_and_scalars["L"]
    M = B * L
    input_dim = 4096
    hidden_size_padded = 2816
    sf_k = input_dim // 16
    
    scaler = BlockWiseScalerNVFP4()
    
    # Generate random input activations
    hidden_states = torch.randn(M, input_dim, dtype=torch.float32, device=device)
    
    # Generate and quantize random weights
    weight_fp32 = torch.randn(1, hidden_size_padded, input_dim, dtype=torch.float32, device=device) * 0.02
    weight_fp4, weight_scale_fp8, weight_global_decode_scale = scaler.quantize(weight_fp32)
    
    # Remove batch dimension
    weight_fp4 = weight_fp4.squeeze(0)
    weight_scale_fp8 = weight_scale_fp8.squeeze(0)
    
    # Convert global_decode_scale tensor to Python float scalar
    weight_global_decode_scale_scalar = float(weight_global_decode_scale.item())
    
    return {
        "hidden_states": hidden_states,
        "weight_fp4": weight_fp4,
        "weight_scale_fp8": weight_scale_fp8,
        "weight_global_decode_scale": weight_global_decode_scale_scalar,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale_fp8: torch.Tensor,
    weight_global_decode_scale: float,
):
    """FP4 block-scaled attention output projection.
    
    Args:
        hidden_states: Input activations (M, input_dim) in float32
        weight_fp4: FP4 quantized weights (hidden_size_padded, input_dim//2)
        weight_scale_fp8: FP8 block scales for weights (hidden_size_padded, sf_k)
        weight_global_decode_scale: Global decode scale for weights (scalar float)
    
    Returns:
        Output tensor (M, hidden_size_padded) in float32
    """
    M, input_dim = hidden_states.shape
    hidden_size_padded = weight_fp4.shape[0]
    device = hidden_states.device
    
    scaler = BlockWiseScalerNVFP4()
    
    # Quantize input activations to FP4
    hidden_states_batched = hidden_states.unsqueeze(0)  # (1, M, input_dim)
    act_fp4, act_scale_fp8, act_global_decode = scaler.quantize(hidden_states_batched)
    
    # Prepare for GEMM: add batch dimension
    act_fp4_batched = act_fp4  # (1, M, input_dim//2)
    act_scale_batched = act_scale_fp8  # (1, M, sf_k)
    weight_fp4_batched = weight_fp4.unsqueeze(0)  # (1, hidden_size_padded, input_dim//2)
    weight_scale_batched = weight_scale_fp8.unsqueeze(0)  # (1, hidden_size_padded, sf_k)
    
    # Convert scalar to tensor for GEMM
    weight_global_decode_tensor = torch.tensor(weight_global_decode_scale, dtype=torch.float32, device=device)
    
    # Create GEMM reference
    gemm_ref = CuBLASRefBlockwiseGemm(
        b=1,
        m=M,
        n=hidden_size_padded,
        k=input_dim
    )
    
    # Perform FP4 block-scaled matrix multiplication
    output = gemm_ref.scaled_mm(
        mat_a=act_fp4_batched,
        mat_b=weight_fp4_batched,
        scale_a=act_scale_batched,
        global_decode_a=act_global_decode,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=weight_scale_batched,
        global_decode_b=weight_global_decode_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32
    )
    
    # Remove batch dimension
    output = output.squeeze(0)  # (M, hidden_size_padded)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.012491 ms |
| - | Scoring Baseline | 0.500000 | 0.357314 ms |
| - | Reference Implementation | 0.159912 | 1.764021 ms |
