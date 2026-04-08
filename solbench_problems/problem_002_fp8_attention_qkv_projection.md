## Description

FP8-quantized QKV projection for multi-head attention with Q/K RMSNorm normalization. Projects hidden states to query, key, and value tensors with blockwise FP8 quantization (1x128 for activations, 128x128 for weights), followed by RMSNorm on Q and K heads. Uses GQA pattern with 60 query heads and 8 key/value heads.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_weight | [q_size, hidden_size] | bfloat16 |
| k_weight | [kv_size, hidden_size] | bfloat16 |
| v_weight | [kv_size, hidden_size] | bfloat16 |
| q_bias | [q_size] | bfloat16 |
| k_bias | [kv_size] | bfloat16 |
| v_bias | [kv_size] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_size | kv_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 16 | 256 | 0.8123 | 0.1690 |
| 2 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 1 | 128 | 0.2921 | 0.0198 |
| 3 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 32 | 256 | 1.2724 | 0.3379 |
| 4 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 2 | 2048 | 0.8027 | 0.1690 |
| 5 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 8 | 1024 | 1.2745 | 0.3379 |
| 6 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 1 | 4096 | 0.8058 | 0.1690 |
| 7 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 2 | 1024 | 0.5512 | 0.0845 |
| 8 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 64 | 128 | 1.3397 | 0.3379 |
| 9 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 4 | 384 | 0.5028 | 0.0634 |
| 10 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 2 | 128 | 0.3023 | 0.0201 |
| 11 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 4 | 128 | 0.3540 | 0.0211 |
| 12 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 1 | 384 | 0.3239 | 0.0204 |
| 13 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 64 | 256 | 2.2473 | 0.6758 |
| 14 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 1 | 1024 | 0.4399 | 0.0422 |
| 15 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 1 | 256 | 0.3024 | 0.0201 |
| 16 | 7680 | 60 | 8 | 128 | 7680 | 1024 | 8 | 128 | 0.4391 | 0.0422 |

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

def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation_scaler: BlockwiseScaler,
    weight_scaler: BlockwiseScaler,
    gemm_ref: CuBLASRefBlockwiseGemm,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """FP8 linear layer with blockwise scaling."""
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    scale_x = activation_scaler.compute_scales(x_fp32)
    weight_t = weight_fp32.T
    scales_w = weight_scaler.compute_scales(weight_t)
    
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    weight_scaled = weight_scaler.apply_scaling(
        weight_t, scales_w, inverse=False, clamp_to_fp8_range=True
    )
    
    qx = x_scaled.to(torch.float8_e4m3fn)
    qw = weight_scaled.T.to(torch.float8_e4m3fn)
    
    scale_w_cublas = scales_w.T.contiguous()
    
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=bias,
        output_dtype=output_dtype,
        use_fast_accum=True,
    )
    
    return output

def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm applied to the last dimension."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """FP8 QKV projection with Q/K RMSNorm normalization."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 60
    num_key_value_heads = 8
    head_dim = 128
    output_dtype = torch.bfloat16
    
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    hidden_states_flat = hidden_states.reshape(-1, hidden_size)
    
    query_states = _fp8_linear(
        hidden_states_flat, q_weight, q_bias,
        activation_scaler, weight_scaler, gemm_ref, output_dtype
    )
    key_states = _fp8_linear(
        hidden_states_flat, k_weight, k_bias,
        activation_scaler, weight_scaler, gemm_ref, output_dtype
    )
    value_states = _fp8_linear(
        hidden_states_flat, v_weight, v_bias,
        activation_scaler, weight_scaler, gemm_ref, output_dtype
    )
    
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    query_states = _rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = _rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    return query_states, key_states, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.081958 ms |
| - | Scoring Baseline | 0.500000 | 0.615157 ms |
| - | Reference Implementation | 0.050204 | 10.671535 ms |
