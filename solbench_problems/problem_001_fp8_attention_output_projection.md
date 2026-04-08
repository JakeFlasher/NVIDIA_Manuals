## Description

FP8-quantized attention output projection that combines multi-head attention outputs back to hidden dimension. Projects concatenated attention outputs from all heads (num_heads * head_dim = 7680) to hidden_size (7680) using blockwise FP8 quantization with BlockWise1x128 for activations and BlockWise128x128 for weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [num_tokens, input_dim] | bfloat16 |
| o_proj_weight | [hidden_size, input_dim] | bfloat16 |
| o_proj_bias | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | head_dim | input_dim | scale_k_act | scale_n | scale_k_weight | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 32 | 256 | 0.8930 | 0.2668 |
| 2 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 16 | 256 | 0.5424 | 0.1334 |
| 3 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 4 | 256 | 0.2775 | 0.0333 |
| 4 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 8 | 2048 | 1.5953 | 0.5335 |
| 5 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 8 | 512 | 0.5448 | 0.1334 |
| 6 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 1 | 256 | 0.1923 | 0.0162 |
| 7 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 16 | 128 | 0.3850 | 0.0667 |
| 8 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 1 | 2048 | 0.3833 | 0.0667 |
| 9 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 1 | 1024 | 0.2762 | 0.0333 |
| 10 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 1 | 512 | 0.2198 | 0.0169 |
| 11 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 8 | 1024 | 0.8718 | 0.2668 |
| 12 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 32 | 128 | 0.5404 | 0.1334 |
| 13 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 2 | 256 | 0.2196 | 0.0169 |
| 14 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 32 | 512 | 1.6042 | 0.5335 |
| 15 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 1 | 4096 | 0.5412 | 0.1334 |
| 16 | 7680 | 60 | 128 | 7680 | 60 | 60 | 60 | 4 | 2048 | 0.8739 | 0.2668 |

```python
import torch
# --- inlined fp8_reference ---
import torch

from enum import StrEnum

class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - TensorWise: Global per-tensor scaling (no blocks)
    - RowWise: Per-row scaling (1 scale per row)
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K)
    - BlockWise1x32: 1x32 blocks (per-tensor in M, 32-sized blocks in K)
    - BlockWise1x128: 1x128 blocks (per-tensor in M, 128-sized blocks in K)
    - BlockWise128x128: 128x128 blocks (blockwise in both dimensions)
    """

    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self) -> tuple[int, int]:
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]

class BlockwiseScaler:
    """
    Compute and apply scales for FP8 tensors.

    Supports various scaling strategies via ScalingType enum:
    - TensorWise: Global per-tensor scaling
    - RowWise: Per-row scaling
    - BlockWise1x16/32/128: Rectangular blocks
    - BlockWise128x128: Square blocks
    """

    E4M3_MAX = 448.0  # Maximum representable value in E4M3

    def __init__(self, scaling_type: ScalingType):
        """
        Initialize BlockwiseScaler with a specific scaling strategy.

        Args:
            scaling_type: ScalingType enum value
                Examples:
                - ScalingType.TensorWise -> global per-tensor scaling
                - ScalingType.RowWise -> per-row scaling (1 scale per row)
                - ScalingType.BlockWise1x128 -> 1x128 blocks
                - ScalingType.BlockWise128x128 -> 128x128 blocks
        """
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape

        # Map enum to block dimensions (M, K)
        scaling_map = {
            ScalingType.TensorWise: (None, None),  # No blocking
            ScalingType.RowWise: (1, None),  # Per-row, full K dimension
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }

        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

        # Keep for backward compatibility (use first dimension if available)
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factors based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Returns scalar tensor
        - RowWise: Returns (M,) tensor
        - BlockWise*: Returns (M//block_size_m, K//block_size_k) tensor

        Args:
            tensor: Input tensor (typically M, K for 2D)

        Returns:
            Scale tensor with shape depending on scaling type.
            These are inverse scales (amax / dtype_max) used for dequantization.
        """
        if self.scaling_type == ScalingType.TensorWise:
            # Global per-tensor scaling
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            # Per-row scaling: (M, K) -> (M,)
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        # BlockWise scaling
        assert M % self.block_size_m == 0, (
            f"M={M} must be a multiple of {self.block_size_m}"
        )
        assert K % self.block_size_k == 0, (
            f"K={K} must be a multiple of {self.block_size_k}"
        )

        # Reshape (M, K) -> (M//block_size_m, block_size_m, K//block_size_k, block_size_k)
        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)

        # Compute max over the block dimensions (dims 1 and 3)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

        # Compute inverse scales
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        """
        Apply scaling to tensor based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Uses scalar scale
        - RowWise: Uses per-row scales (M,)
        - BlockWise*: Uses blockwise scales (M//block_size_m, K//block_size_k)

        Args:
            tensor: Input tensor (typically M, K for 2D)
            scales: Scale tensor with shape depending on scaling type
                   These are inverse scales (amax / dtype_max)
            inverse: If True, multiply by scales (dequantization)
                    If False, divide by scales (quantization)
            clamp_to_fp8_range: If True, clamp to FP8 range before returning

        Returns:
            Scaled tensor (same shape as input)
        """
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            # expand (M,) -> (M, 1)
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            # blockwise scaling
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)

        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(
                    tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX
                )

        return tensor_scaled.reshape(*old_shape)

class CuBLASRefBlockwiseGemm:
    """
    Reference implementation of blockwise-scaled GEMM via dequantize-then-matmul.
    """

    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        """
        Scaled matrix multiplication: dequantize A and B, then matmul in float32.

        Args:
            mat_a: Input matrix A (M, K) in float8_e4m3fn
            mat_b: Input matrix B (N, K) in float8_e4m3fn
            scale_a: Scaling factors for A
            scale_recipe_a: Scaling type for A
            scale_b: Scaling factors for B
            scale_recipe_b: Scaling type for B
            bias: Optional bias vector (N,)
            output_dtype: Output data type
            use_fast_accum: Unused (kept for API compatibility)

        Returns:
            Result matrix (M, N) with dtype=output_dtype
        """
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        # Dequantize: FP8 values * inverse_scales -> float32
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        # Single matmul in float32
        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)

# --- end inlined fp8_reference ---

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor,
):
    """
    FP8-quantized attention output projection.
    
    Args:
        attn_output: Concatenated attention outputs (num_tokens, input_dim)
        o_proj_weight: Output projection weight (hidden_size, input_dim)
        o_proj_bias: Output projection bias (hidden_size)
    
    Returns:
        Projected output (num_tokens, hidden_size)
    """
    M, K = attn_output.shape
    N = o_proj_weight.shape[0]
    
    # Initialize scalers
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Step 1: Compute activation scales
    attn_output_fp32 = attn_output.to(torch.float32)
    scale_x = activation_scaler.compute_scales(attn_output_fp32)
    
    # Step 2: Apply scaling and quantize activations
    attn_output_scaled = activation_scaler.apply_scaling(
        attn_output_fp32,
        scale_x,
        inverse=False,
        clamp_to_fp8_range=True
    )
    qx = attn_output_scaled.to(torch.float8_e4m3fn)
    
    # Step 3: Compute weight scales and quantize weights
    weight_fp32 = o_proj_weight.to(torch.float32)
    weight_t = weight_fp32.T  # (K, N) for blockwise scaling
    scales_w = weight_scaler.compute_scales(weight_t)  # (K//128, N//128)
    
    weight_scaled = weight_scaler.apply_scaling(
        weight_t,
        scales_w,
        inverse=False,
        clamp_to_fp8_range=True
    )
    qw = weight_scaled.T.to(torch.float8_e4m3fn)  # (N, K)
    
    # Transpose scales to CuBLAS format: (N//128, K//128)
    scale_w_cublas = scales_w.T.contiguous()
    
    # Step 4: Perform FP8 GEMM
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=o_proj_bias,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.094308 ms |
| - | Scoring Baseline | 0.500000 | 0.501088 ms |
| - | Reference Implementation | 0.040343 | 10.052536 ms |
