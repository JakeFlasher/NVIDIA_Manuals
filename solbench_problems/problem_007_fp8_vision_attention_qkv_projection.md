## Description

FP8-quantized QKV projection for vision transformer attention. Projects vision hidden states to query, key, and value tensors using a single fused linear layer with blockwise FP8 quantization (BlockWise1x128 for activations, BlockWise128x128 for weights).
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [seq_len, hidden_size] | bfloat16 |
| qkv_weight | [qkv_out_features, hidden_size] | bfloat16 |
| qkv_bias | [qkv_out_features] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [seq_len, num_heads, head_dim] | bfloat16 |
| key_states | [seq_len, num_heads, head_dim] | bfloat16 |
| value_states | [seq_len, num_heads, head_dim] | bfloat16 |

| # | hidden_size | num_heads | head_dim | qkv_out_features | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 16 | 96 | 4608 | 1536 | 0.1230 | 0.0060 |
| 2 | 1536 | 16 | 96 | 4608 | 1152 | 0.1165 | 0.0045 |
| 3 | 1536 | 16 | 96 | 4608 | 128 | 0.0981 | 0.0019 |
| 4 | 1536 | 16 | 96 | 4608 | 2688 | 0.1473 | 0.0105 |
| 5 | 1536 | 16 | 96 | 4608 | 3712 | 0.1661 | 0.0145 |
| 6 | 1536 | 16 | 96 | 4608 | 384 | 0.1051 | 0.0022 |
| 7 | 1536 | 16 | 96 | 4608 | 512 | 0.1042 | 0.0023 |
| 8 | 1536 | 16 | 96 | 4608 | 3200 | 0.1521 | 0.0125 |
| 9 | 1536 | 16 | 96 | 4608 | 2944 | 0.1481 | 0.0115 |
| 10 | 1536 | 16 | 96 | 4608 | 896 | 0.1124 | 0.0035 |
| 11 | 1536 | 16 | 96 | 4608 | 3840 | 0.1652 | 0.0150 |
| 12 | 1536 | 16 | 96 | 4608 | 2176 | 0.1343 | 0.0085 |
| 13 | 1536 | 16 | 96 | 4608 | 1664 | 0.1276 | 0.0065 |
| 14 | 1536 | 16 | 96 | 4608 | 1024 | 0.1111 | 0.0040 |
| 15 | 1536 | 16 | 96 | 4608 | 3456 | 0.1590 | 0.0135 |
| 16 | 1536 | 16 | 96 | 4608 | 768 | 0.1072 | 0.0030 |

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
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    """
    FP8-quantized QKV projection for vision transformer attention.
    
    Args:
        hidden_states: Input tensor (seq_len, hidden_size)
        qkv_weight: QKV projection weight (qkv_out_features, hidden_size)
        qkv_bias: QKV projection bias (qkv_out_features,)
    
    Returns:
        Tuple of (query_states, key_states, value_states)
    """
    # Constants
    num_heads = 16
    head_dim = 96
    
    seq_length = hidden_states.shape[0]
    
    # Scaling types
    activation_scaling_type = ScalingType.BlockWise1x128
    weight_scaling_type = ScalingType.BlockWise128x128
    
    # Create scalers
    activation_scaler = BlockwiseScaler(activation_scaling_type)
    weight_scaler = BlockwiseScaler(weight_scaling_type)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Convert to FP32 for scaling operations
    hidden_states_fp32 = hidden_states.to(torch.float32)
    
    # Step 1: Compute activation scales dynamically
    scale_x = activation_scaler.compute_scales(hidden_states_fp32)
    
    # Step 2: Apply scaling before quantization (for activations)
    hidden_states_scaled = activation_scaler.apply_scaling(
        hidden_states_fp32,
        scale_x,
        inverse=False,
        clamp_to_fp8_range=True
    )
    
    # Step 3: Quantize activation to FP8
    qx = hidden_states_scaled.to(torch.float8_e4m3fn)  # (seq_len, hidden_size)
    
    # Step 4: Compute weight scales and quantize weight to FP8
    w_fp32 = qkv_weight.T.to(torch.float32)  # (hidden_size, qkv_out_features)
    weight_scales = weight_scaler.compute_scales(w_fp32)
    
    w_scaled = weight_scaler.apply_scaling(
        w_fp32,
        weight_scales,
        inverse=False,
        clamp_to_fp8_range=True
    )
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # (qkv_out_features, hidden_size)
    
    # Step 5: Transpose weight scales for CuBLAS format
    scale_w_cublas = weight_scales.T.contiguous()
    
    # Step 6: Perform FP8 GEMM using CuBLAS reference
    qkv = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=activation_scaling_type,
        scale_b=scale_w_cublas,
        scale_recipe_b=weight_scaling_type,
        bias=qkv_bias,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Step 7: Reshape and split into Q, K, V
    # qkv shape: (seq_len, 3 * num_heads * head_dim)
    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    
    # Ensure contiguous output
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    
    return query_states, key_states, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.005976 ms |
| - | Scoring Baseline | 0.500000 | 0.127866 ms |
| - | Reference Implementation | 0.117265 | 0.956968 ms |
