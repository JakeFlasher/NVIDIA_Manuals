## Description

FP8-quantized attention output projection (o_proj) that projects concatenated attention outputs from (num_heads * v_head_dim) = 16384 dimensions back to hidden_size = 7168. Uses BlockWise1x128 scaling for activations and BlockWise128x128 scaling for weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_seq_len, input_dim] | bfloat16 |
| o_proj_weight | [hidden_size, input_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | input_dim | block_size | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 16384 | 128 | 3712 | 1.5846 | 0.2668 |
| 2 | 7168 | 16384 | 128 | 1792 | 0.9876 | 0.1334 |
| 3 | 7168 | 16384 | 128 | 4096 | 1.7785 | 0.0333 |
| 4 | 7168 | 16384 | 128 | 3328 | 1.5408 | 0.5335 |
| 5 | 7168 | 16384 | 128 | 3072 | 1.4060 | 0.1334 |
| 6 | 7168 | 16384 | 128 | 1152 | 0.7835 | 0.0162 |
| 7 | 7168 | 16384 | 128 | 1408 | 0.9629 | 0.0667 |
| 8 | 7168 | 16384 | 128 | 2816 | 1.3655 | 0.0667 |
| 9 | 7168 | 16384 | 128 | 2944 | 1.3832 | 0.0333 |
| 10 | 7168 | 16384 | 128 | 3200 | 1.3830 | 0.0169 |
| 11 | 7168 | 16384 | 128 | 1664 | 0.9737 | 0.2668 |
| 12 | 7168 | 16384 | 128 | 512 | 0.5680 | 0.1334 |
| 13 | 7168 | 16384 | 128 | 3840 | 1.5927 | 0.0169 |
| 14 | 7168 | 16384 | 128 | 2560 | 1.1914 | 0.5335 |
| 15 | 7168 | 16384 | 128 | 2304 | 1.1860 | 0.1334 |
| 16 | 7168 | 16384 | 128 | 2176 | 1.1850 | 0.2668 |

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

def get_inputs(axes_and_scalars: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    input_dim = axes_and_scalars["input_dim"]
    dtype = torch.bfloat16
    
    attn_output = torch.randn(batch_seq_len, input_dim, dtype=dtype, device=device)
    o_proj_weight = torch.randn(hidden_size, input_dim, dtype=dtype, device=device)
    
    return {
        "attn_output": attn_output,
        "o_proj_weight": o_proj_weight,
    }

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 attention output projection with blockwise scaling.
    
    Performs: output = attn_output @ o_proj_weight.T
    
    Scaling approach:
    - attn_output: BlockWise1x128 (per-row with 128-sized blocks in K)
    - o_proj_weight: BlockWise128x128 (128x128 blocks)
    
    Args:
        attn_output: Input tensor [batch_seq_len, input_dim]
        o_proj_weight: Weight matrix [hidden_size, input_dim]
    
    Returns:
        output: Projected tensor [batch_seq_len, hidden_size]
    """
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Convert to FP32 for scale computation
    attn_output_fp32 = attn_output.to(torch.float32)
    
    # Compute activation scales: BlockWise1x128 -> shape (M, K//128)
    scale_input = activation_scaler.compute_scales(attn_output_fp32)
    
    # Weight scales: BlockWise128x128
    # Transpose weight for scale computation: [input_dim, hidden_size]
    weight_fp32 = o_proj_weight.T.to(torch.float32)
    scale_weight = weight_scaler.compute_scales(weight_fp32)
    
    # Apply scaling and quantize to FP8
    attn_output_scaled = activation_scaler.apply_scaling(
        attn_output_fp32,
        scale_input,
        inverse=False,
        clamp_to_fp8_range=True
    )
    attn_output_fp8 = attn_output_scaled.to(torch.float8_e4m3fn)
    
    # Weight scaling and quantization
    weight_scaled = weight_scaler.apply_scaling(
        weight_fp32,
        scale_weight,
        inverse=False,
        clamp_to_fp8_range=True
    )
    # Transpose back to [hidden_size, input_dim] for matmul
    weight_fp8 = weight_scaled.T.to(torch.float8_e4m3fn)
    
    # Transpose weight scales to match CuBLAS format
    # From (K//128, N//128) to (N//128, K//128)
    scale_weight_cublas = scale_weight.T.contiguous()
    
    # Perform FP8 GEMM using CuBLAS reference
    output = gemm_ref.scaled_mm(
        mat_a=attn_output_fp8,
        mat_b=weight_fp8,
        scale_a=scale_input,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_weight_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    return output

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"batch_seq_len": 4096, "hidden_size": 7168, "input_dim": 16384},
        device=torch.device("cuda:0")
    )
    out = run(**inputs)
    print(out.shape)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.094308 ms |
| - | Scoring Baseline | 0.500000 | 1.196010 ms |
| - | Reference Implementation | 0.197114 | 4.360629 ms |
