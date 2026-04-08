## Description

FP4-quantized linear layer with block-wise scaling (16 elements per block). Computes y = x @ W^T where both x and W are quantized to FP4 with FP8 E4M3FN block-wise scale factors. Uses torch._scaled_mm for efficient FP4 matrix multiplication.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, in_features] | bfloat16 |
| weight_fp4 | [out_features, in_features_half] | float4_e2m1fn_x2 |
| weight_scales | [out_features, sf_k] | float8_e4m3fn |
| weight_global_decode_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, out_features] | bfloat16 |

| # | in_features | out_features | sf_vec_size | in_features_half | sf_k | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 8192 | 16 | 4096 | 512 | 8 | 256 | 2.8624 | 0.0213 |
| 2 | 8192 | 8192 | 16 | 4096 | 512 | 1 | 128 | 2.0385 | 0.0013 |
| 3 | 8192 | 8192 | 16 | 4096 | 512 | 16 | 128 | 3.0864 | 0.0213 |
| 4 | 8192 | 8192 | 16 | 4096 | 512 | 4 | 1024 | 3.6016 | 0.0427 |
| 5 | 8192 | 8192 | 16 | 4096 | 512 | 32 | 256 | 4.4715 | 0.0853 |
| 6 | 8192 | 8192 | 16 | 4096 | 512 | 4 | 128 | 2.5931 | 0.0053 |
| 7 | 8192 | 8192 | 16 | 4096 | 512 | 1 | 2048 | 2.8282 | 0.0213 |
| 8 | 8192 | 8192 | 16 | 4096 | 512 | 2 | 1024 | 2.7075 | 0.0213 |
| 9 | 8192 | 8192 | 16 | 4096 | 512 | 2 | 128 | 2.3134 | 0.0027 |
| 10 | 8192 | 8192 | 16 | 4096 | 512 | 4 | 2048 | 4.5856 | 0.0853 |
| 11 | 8192 | 8192 | 16 | 4096 | 512 | 1 | 512 | 2.4914 | 0.0053 |
| 12 | 8192 | 8192 | 16 | 4096 | 512 | 1 | 1024 | 2.7079 | 0.0107 |
| 13 | 8192 | 8192 | 16 | 4096 | 512 | 64 | 256 | 8.2428 | 0.1707 |
| 14 | 8192 | 8192 | 16 | 4096 | 512 | 8 | 128 | 2.8365 | 0.0107 |
| 15 | 8192 | 8192 | 16 | 4096 | 512 | 8 | 1024 | 4.5117 | 0.0853 |
| 16 | 8192 | 8192 | 16 | 4096 | 512 | 2 | 512 | 2.8386 | 0.0107 |

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

def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs for FP4 linear layer."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    in_features = axes_and_scalars["in_features"]
    out_features = axes_and_scalars["out_features"]
    
    scaler = BlockWiseScalerNVFP4()
    
    # Input tensor in bfloat16
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16, device=device)
    
    # Generate FP32 weight and quantize to FP4
    weight_fp32 = torch.randn(1, out_features, in_features, dtype=torch.float32, device=device)
    weight_fp4, weight_scales, weight_global_decode_scale = scaler.quantize(weight_fp32)
    
    # Remove batch dimension from weight tensors
    weight_fp4 = weight_fp4.squeeze(0)  # [out_features, in_features//2]
    weight_scales = weight_scales.squeeze(0)  # [out_features, sf_k]
    
    # Extract scalar value from tensor
    weight_global_decode_scale_scalar = weight_global_decode_scale.item()
    
    return {
        "x": x,
        "weight_fp4": weight_fp4,
        "weight_scales": weight_scales,
        "weight_global_decode_scale": weight_global_decode_scale_scalar,
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_global_decode_scale: float,
):
    """FP4 linear layer forward pass.
    
    Args:
        x: Input tensor [batch_size, seq_len, in_features] in bfloat16
        weight_fp4: FP4 quantized weights [out_features, in_features//2]
        weight_scales: Block-wise scales [out_features, sf_k] in FP8 E4M3FN
        weight_global_decode_scale: Global decode scale (scalar float)
    
    Returns:
        Output tensor [batch_size, seq_len, out_features] in bfloat16
    """
    batch_size, seq_len, in_features = x.shape
    out_features = weight_fp4.shape[0]
    
    scaler = BlockWiseScalerNVFP4()
    
    # Reshape input for matrix multiplication
    x_flat = x.reshape(-1, in_features).to(torch.float32)  # [batch*seq, in_features]
    
    # Quantize input to FP4 with block-wise scaling
    x_batched = x_flat.unsqueeze(0)  # [1, batch*seq, in_features]
    x_fp4, x_scales, x_global_decode = scaler.quantize(x_batched)
    
    # Remove batch dimension
    x_fp4 = x_fp4.squeeze(0)  # [batch*seq, in_features//2]
    x_scales = x_scales.squeeze(0)  # [batch*seq, sf_k]
    
    # Convert scales to blocked format for torch._scaled_mm
    x_scales_blocked = scaler.convert_to_blocked_format_for_pytorch_scaled_mm(x_scales)
    weight_scales_blocked = scaler.convert_to_blocked_format_for_pytorch_scaled_mm(weight_scales)
    
    # Perform FP4 matrix multiplication
    # x_fp4: [batch*seq, in_features//2]
    # weight_fp4: [out_features, in_features//2]
    # We need: [batch*seq, in_features//2] @ [in_features//2, out_features] -> [batch*seq, out_features]
    output = torch._scaled_mm(
        x_fp4,
        weight_fp4.transpose(0, 1),  # [in_features//2, out_features]
        x_scales_blocked,
        weight_scales_blocked,
        bias=None,
        out_dtype=torch.bfloat16,
    )
    
    # Apply global decode scales
    global_scale = x_global_decode.item() * weight_global_decode_scale
    output = output * global_scale
    
    # Reshape back to [batch_size, seq_len, out_features]
    output = output.reshape(batch_size, seq_len, out_features)
    
    return output

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={
            "batch_size": 2,
            "seq_len": 256,
            "in_features": 8192,
            "out_features": 8192,
            "sf_vec_size": 16,
        },
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(out.shape)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.017940 ms |
| - | Scoring Baseline | 0.500000 | 3.204131 ms |
| - | Reference Implementation | 0.183525 | 14.224450 ms |
