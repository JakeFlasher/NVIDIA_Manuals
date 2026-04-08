## Description

FP8-quantized vision attention output projection that combines multi-head attention outputs back to hidden dimension. Projects concatenated attention outputs from all heads through a linear layer with blockwise FP8 quantization (1x128 for activation, 128x128 for weight).
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [seq_len, hidden_size] | bfloat16 |
| weight | [hidden_size, hidden_size] | bfloat16 |
| scale_attn | [seq_len, num_hidden_blocks] | float32 |
| scale_weight | [num_hidden_blocks, num_hidden_blocks] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_hidden_blocks | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 10 | 1152 | 0.1165 | 0.0010 |
| 2 | 1280 | 10 | 3840 | 0.1544 | 0.0035 |
| 3 | 1280 | 10 | 3712 | 0.1552 | 0.0034 |
| 4 | 1280 | 10 | 2688 | 0.1354 | 0.0024 |
| 5 | 1280 | 10 | 1920 | 0.1270 | 0.0017 |
| 6 | 1280 | 10 | 256 | 0.1074 | 0.0005 |
| 7 | 1280 | 10 | 3456 | 0.1493 | 0.0031 |
| 8 | 1280 | 10 | 128 | 0.1066 | 0.0005 |
| 9 | 1280 | 10 | 2816 | 0.1400 | 0.0025 |
| 10 | 1280 | 10 | 1408 | 0.1183 | 0.0013 |
| 11 | 1280 | 10 | 2560 | 0.1347 | 0.0023 |
| 12 | 1280 | 10 | 3584 | 0.1479 | 0.0032 |
| 13 | 1280 | 10 | 3200 | 0.1422 | 0.0029 |
| 14 | 1280 | 10 | 768 | 0.1086 | 0.0007 |
| 15 | 1280 | 10 | 640 | 0.1087 | 0.0006 |
| 16 | 1280 | 10 | 1664 | 0.1185 | 0.0015 |

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

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Generate random inputs and compute scales for FP8 vision attention output projection.
    
    Returns:
        Dict of (attn_output, weight, scale_attn, scale_weight)
    """
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    
    attn_output = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Convert to FP32 for scale computation
    attn_output_fp32 = attn_output.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Transpose weight for blockwise scale computation
    weight_fp32_t = weight_fp32.T  # (K, N)
    
    # Compute scales using unified interface
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    scale_attn = activation_scaler.compute_scales(attn_output_fp32)
    
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    scale_weight = weight_scaler.compute_scales(weight_fp32_t)
    
    return {
        "attn_output": attn_output,
        "weight": weight,
        "scale_attn": scale_attn,
        "scale_weight": scale_weight,
    }

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor,
):
    """
    FP8 vision attention output projection with blockwise scaling.
    
    Scaling approach:
    - attn_output (input): BlockWise1x128 scaling
    - weight: BlockWise128x128 scaling
    
    Args:
        attn_output: Input tensor (seq_len, hidden_size) in BF16
        weight: Weight tensor (hidden_size, hidden_size) in BF16
        scale_attn: Blockwise scales for attn_output (seq_len, hidden_size//128)
        scale_weight: Blockwise scales for weight (hidden_size//128, hidden_size//128)
    
    Returns:
        Output tensor (seq_len, hidden_size)
    """
    M, K = attn_output.shape
    N, K_w = weight.shape
    assert K == K_w, f"K dimension mismatch: {K} vs {K_w}"
    
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Convert to FP32 for scaling operations
    x_fp32 = attn_output.to(torch.float32)
    w_fp32 = weight.to(torch.float32)
    
    # Step 1: Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_attn, inverse=False, clamp_to_fp8_range=True
    )
    
    # Transpose w to (K, N) for blockwise scaling
    w_fp32_t = w_fp32.T  # (K, N)
    w_scaled = weight_scaler.apply_scaling(
        w_fp32_t, scale_weight, inverse=False, clamp_to_fp8_range=True
    )
    
    # Step 2: Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # (M, K)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # (N, K) - transpose back
    
    # Step 3: Transpose weight scales to match CuBLAS format
    # Scales from compute_scales are (K//128, N//128), need (N//128, K//128)
    scale_w_cublas = scale_weight.T.contiguous()  # (N//128, K//128)
    
    # Step 4: Call CuBLAS reference implementation
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_attn,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=attn_output.dtype,
        use_fast_accum=True,
    )
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001599 ms |
| - | Scoring Baseline | 0.500000 | 0.128294 ms |
| - | Reference Implementation | 0.293704 | 0.308746 ms |
