## Description

FP4-quantized fused MLP with SiLU activation and gating mechanism. Implements down_proj(SiLU(gate_proj(x)) * up_proj(x)) with FP4 quantization and block-wise scaling (16 elements per block). Gate and up projections share input quantization, followed by fused SiLU gating, then down projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate_weight_fp4 | [1, intermediate_size, hidden_half] | float4_e2m1fn_x2 |
| gate_scale | [1, intermediate_size, hidden_scale_blocks] | float8_e4m3fn |
| gate_global_decode_scale | scalar | float32 |
| up_weight_fp4 | [1, intermediate_size, hidden_half] | float4_e2m1fn_x2 |
| up_scale | [1, intermediate_size, hidden_scale_blocks] | float8_e4m3fn |
| up_global_decode_scale | scalar | float32 |
| down_weight_fp4 | [1, hidden_size, intermediate_half] | float4_e2m1fn_x2 |
| down_scale | [1, hidden_size, intermediate_scale_blocks] | float8_e4m3fn |
| down_global_decode_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | hidden_half | intermediate_half | hidden_scale_blocks | intermediate_scale_blocks | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 4 | 256 | 0.9916 | 0.0232 |
| 2 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 2 | 1024 | 1.3666 | 0.0465 |
| 3 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 8 | 128 | 0.9887 | 0.0232 |
| 4 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 1 | 4096 | 2.1024 | 0.0930 |
| 5 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 8 | 2048 | 6.9343 | 0.3720 |
| 6 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 2 | 512 | 1.0029 | 0.0232 |
| 7 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 1 | 512 | 0.9299 | 0.0116 |
| 8 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 1 | 128 | 0.9113 | 0.0029 |
| 9 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 8 | 512 | 2.0986 | 0.0930 |
| 10 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 1 | 256 | 0.9214 | 0.0058 |
| 11 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 4 | 2048 | 3.7025 | 0.1860 |
| 12 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 1 | 2048 | 1.3669 | 0.0465 |
| 13 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 32 | 128 | 2.1063 | 0.0930 |
| 14 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 32 | 512 | 6.9357 | 0.3720 |
| 15 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 4 | 128 | 0.9325 | 0.0116 |
| 16 | 4096 | 12288 | 2048 | 6144 | 256 | 768 | 2 | 128 | 0.9384 | 0.0058 |

```python
import torch
import torch.nn.functional as F
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
    """Generate inputs for FP4 fused MLP with SiLU gating."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 4096
    intermediate_size = 12288
    
    scaler = BlockWiseScalerNVFP4()
    
    # Input activation
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Gate projection weights
    gate_weight_fp32 = torch.randn(1, intermediate_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    gate_fp4, gate_scale, gate_global = scaler.quantize(gate_weight_fp32)
    
    # Up projection weights
    up_weight_fp32 = torch.randn(1, intermediate_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    up_fp4, up_scale, up_global = scaler.quantize(up_weight_fp32)
    
    # Down projection weights
    down_weight_fp32 = torch.randn(1, hidden_size, intermediate_size, dtype=torch.float32, device=device) * 0.02
    down_fp4, down_scale, down_global = scaler.quantize(down_weight_fp32)
    
    return {
        "x": x,
        "gate_weight_fp4": gate_fp4,
        "gate_scale": gate_scale,
        "gate_global_decode_scale": gate_global.item(),
        "up_weight_fp4": up_fp4,
        "up_scale": up_scale,
        "up_global_decode_scale": up_global.item(),
        "down_weight_fp4": down_fp4,
        "down_scale": down_scale,
        "down_global_decode_scale": down_global.item(),
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    gate_weight_fp4: torch.Tensor,
    gate_scale: torch.Tensor,
    gate_global_decode_scale: float,
    up_weight_fp4: torch.Tensor,
    up_scale: torch.Tensor,
    up_global_decode_scale: float,
    down_weight_fp4: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_decode_scale: float,
):
    """FP4 fused MLP with SiLU gating: down_proj(SiLU(gate_proj(x)) * up_proj(x))"""
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = gate_weight_fp4.shape[1]
    
    scaler = BlockWiseScalerNVFP4()
    
    # Flatten batch and sequence dimensions
    x_flat = x.reshape(-1, hidden_size).contiguous()
    effective_batch = x_flat.shape[0]
    
    # Pad effective_batch to multiple of 128 for GEMM requirements
    pad_size = (128 - effective_batch % 128) % 128
    if pad_size > 0:
        x_flat = F.pad(x_flat, (0, 0, 0, pad_size))
        effective_batch_padded = x_flat.shape[0]
    else:
        effective_batch_padded = effective_batch
    
    # Convert scalar scales to tensors
    gate_global_tensor = torch.tensor(gate_global_decode_scale, dtype=torch.float32, device=x.device)
    up_global_tensor = torch.tensor(up_global_decode_scale, dtype=torch.float32, device=x.device)
    down_global_tensor = torch.tensor(down_global_decode_scale, dtype=torch.float32, device=x.device)
    
    # Quantize input activation to FP4
    x_fp4, x_scale, x_global_decode = scaler.quantize(x_flat.unsqueeze(0).to(torch.float32))
    
    # Gate projection GEMM
    gate_gemm = CuBLASRefBlockwiseGemm(b=1, m=effective_batch_padded, n=intermediate_size, k=hidden_size)
    
    gate_out = gate_gemm.scaled_mm(
        mat_a=x_fp4,
        mat_b=gate_weight_fp4,
        scale_a=x_scale,
        global_decode_a=x_global_decode,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=gate_scale,
        global_decode_b=gate_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    
    # Up projection GEMM (reuse input quantization)
    up_out = gate_gemm.scaled_mm(
        mat_a=x_fp4,
        mat_b=up_weight_fp4,
        scale_a=x_scale,
        global_decode_a=x_global_decode,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=up_scale,
        global_decode_b=up_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    
    # Fused SiLU activation and gating
    gate_out = gate_out.squeeze(0)
    up_out = up_out.squeeze(0)
    hidden = F.silu(gate_out) * up_out
    
    # Quantize hidden for down projection
    hidden_fp4, hidden_scale, hidden_global_decode = scaler.quantize(hidden.unsqueeze(0).to(torch.float32))
    
    # Down projection GEMM
    down_gemm = CuBLASRefBlockwiseGemm(b=1, m=effective_batch_padded, n=hidden_size, k=intermediate_size)
    
    output = down_gemm.scaled_mm(
        mat_a=hidden_fp4,
        mat_b=down_weight_fp4,
        scale_a=hidden_scale,
        global_decode_a=hidden_global_decode,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=down_scale,
        global_decode_b=down_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    
    # Remove padding and reshape back to original dimensions
    output = output.squeeze(0)[:effective_batch - pad_size + pad_size if pad_size == 0 else effective_batch]
    output = output[:batch_size * seq_len].reshape(batch_size, seq_len, hidden_size)
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.035855 ms |
| - | Scoring Baseline | 0.500000 | 1.611896 ms |
| - | Reference Implementation | 0.276486 | 4.212366 ms |
