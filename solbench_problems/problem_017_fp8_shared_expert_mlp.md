## Description

FP8-quantized shared expert MLP for DeepSeek-V3 MoE. Processes ALL tokens unconditionally using SwiGLU activation: output = down_proj(silu(gate_proj(x)) * up_proj(x)). Uses blockwise FP8 quantization with 128x128 block scaling for weights and per-row scaling for activations.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [num_tokens, hidden_size] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| gate_proj_weight_scales | [hidden_blocks, intermediate_blocks] | float32 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight_scales | [hidden_blocks, intermediate_blocks] | float32 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| down_proj_weight_scales | [intermediate_blocks, hidden_blocks] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | block_size | hidden_blocks | intermediate_blocks | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 2048 | 128 | 56 | 16 | 1280 | 0.9080 | 0.0685 |
| 2 | 7168 | 2048 | 128 | 56 | 16 | 3072 | 1.2215 | 0.0934 |
| 3 | 7168 | 2048 | 128 | 56 | 16 | 2176 | 1.0187 | 0.0156 |
| 4 | 7168 | 2048 | 128 | 56 | 16 | 3712 | 1.3811 | 0.0654 |
| 5 | 7168 | 2048 | 128 | 56 | 16 | 2816 | 1.1913 | 0.0809 |
| 6 | 7168 | 2048 | 128 | 56 | 16 | 1920 | 0.9853 | 0.0996 |
| 7 | 7168 | 2048 | 128 | 56 | 16 | 3456 | 1.3629 | 0.0591 |
| 8 | 7168 | 2048 | 128 | 56 | 16 | 384 | 0.7375 | 0.0122 |
| 9 | 7168 | 2048 | 128 | 56 | 16 | 512 | 0.7435 | 0.0342 |
| 10 | 7168 | 2048 | 128 | 56 | 16 | 3584 | 1.3743 | 0.0187 |
| 11 | 7168 | 2048 | 128 | 56 | 16 | 128 | 0.6883 | 0.0622 |
| 12 | 7168 | 2048 | 128 | 56 | 16 | 3200 | 1.2345 | 0.0965 |
| 13 | 7168 | 2048 | 128 | 56 | 16 | 2432 | 1.1285 | 0.0129 |
| 14 | 7168 | 2048 | 128 | 56 | 16 | 1408 | 0.9297 | 0.0716 |
| 15 | 7168 | 2048 | 128 | 56 | 16 | 1792 | 0.9951 | 0.0560 |
| 16 | 7168 | 2048 | 128 | 56 | 16 | 2304 | 1.0388 | 0.0218 |

```python
import torch
import torch.nn.functional as F
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

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with proper FP8 weight scales."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    
    # Input hidden states
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Weight matrices - scaled by 1/sqrt(fan_in) to keep output magnitudes reasonable
    gate_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    up_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    down_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * (intermediate_size ** -0.5)
    
    # Compute weight scales using BlockwiseScaler
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    # Gate projection: weight is [N, K] = [intermediate_size, hidden_size]
    # For GEMM: x @ weight.T, we need scales for weight.T which is [K, N] = [hidden_size, intermediate_size]
    gate_weight_t = gate_proj_weight.T.to(torch.float32)  # [hidden_size, intermediate_size]
    gate_proj_weight_scales = weight_scaler.compute_scales(gate_weight_t)  # [hidden_size//128, intermediate_size//128]
    
    # Up projection: same shape as gate
    up_weight_t = up_proj_weight.T.to(torch.float32)  # [hidden_size, intermediate_size]
    up_proj_weight_scales = weight_scaler.compute_scales(up_weight_t)  # [hidden_size//128, intermediate_size//128]
    
    # Down projection: weight is [N, K] = [hidden_size, intermediate_size]
    # For GEMM: intermediate @ weight.T, we need scales for weight.T which is [K, N] = [intermediate_size, hidden_size]
    down_weight_t = down_proj_weight.T.to(torch.float32)  # [intermediate_size, hidden_size]
    down_proj_weight_scales = weight_scaler.compute_scales(down_weight_t)  # [intermediate_size//128, hidden_size//128]
    
    return {
        "x": x,
        "gate_proj_weight": gate_proj_weight,
        "gate_proj_weight_scales": gate_proj_weight_scales,
        "up_proj_weight": up_proj_weight,
        "up_proj_weight_scales": up_proj_weight_scales,
        "down_proj_weight": down_proj_weight,
        "down_proj_weight_scales": down_proj_weight_scales,
    }

def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    activation_scaler: BlockwiseScaler,
    weight_scaler: BlockwiseScaler,
    gemm_ref: CuBLASRefBlockwiseGemm,
) -> torch.Tensor:
    """
    FP8 linear layer with blockwise scaling.
    
    Args:
        x: Input tensor [M, K] in BF16
        weight: Weight tensor [N, K] in BF16
        weight_scales: Pre-computed weight scales [K//128, N//128]
    
    Returns:
        Output tensor [M, N] in BF16
    """
    M, K = x.shape
    N, _ = weight.shape
    
    # Step 1: Compute activation scales dynamically
    x_fp32 = x.to(torch.float32)
    scale_x = activation_scaler.compute_scales(x_fp32)
    
    # Step 2: Apply scaling and quantize input
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    qx = x_scaled.to(torch.float8_e4m3fn)  # [M, K]
    
    # Step 3: Apply scaling and quantize weight
    weight_fp32 = weight.T.to(torch.float32)  # [K, N]
    weight_scaled = weight_scaler.apply_scaling(
        weight_fp32, weight_scales, inverse=False, clamp_to_fp8_range=True
    )
    qw = weight_scaled.T.to(torch.float8_e4m3fn)  # [N, K]
    
    # Step 4: Transpose weight scales for CuBLAS format
    # Scales from compute_scales are [K//128, N//128], need [N//128, K//128]
    scale_w_cublas = weight_scales.T.contiguous()
    
    # Step 5: FP8 GEMM using CuBLAS reference
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True
    )
    
    return output

@torch.no_grad()
def run(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_weight_scales: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_weight_scales: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_weight_scales: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 Shared Expert MLP forward pass.
    
    Computation:
        gate = silu(gate_proj(x))  # SiLU NOT quantized
        up = up_proj(x)
        output = down_proj(gate * up)
    
    Args:
        x: Input tensor [num_tokens, hidden_size] in BF16
        gate_proj_weight: Gate projection weights [intermediate_size, hidden_size]
        gate_proj_weight_scales: FP8 scales for gate weights [hidden_size//128, intermediate_size//128]
        up_proj_weight: Up projection weights [intermediate_size, hidden_size]
        up_proj_weight_scales: FP8 scales for up weights [hidden_size//128, intermediate_size//128]
        down_proj_weight: Down projection weights [hidden_size, intermediate_size]
        down_proj_weight_scales: FP8 scales for down weights [intermediate_size//128, hidden_size//128]
    
    Returns:
        Output tensor [num_tokens, hidden_size] in BF16
    """
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # FP8 gate projection: [num_tokens, hidden_size] @ [hidden_size, intermediate_size] -> [num_tokens, intermediate_size]
    gate_output = _fp8_linear(
        x, gate_proj_weight, gate_proj_weight_scales,
        activation_scaler, weight_scaler, gemm_ref
    )
    
    # SiLU activation (NOT quantized, remains in BF16)
    gate_activated = F.silu(gate_output)
    
    # FP8 up projection: [num_tokens, hidden_size] @ [hidden_size, intermediate_size] -> [num_tokens, intermediate_size]
    up_output = _fp8_linear(
        x, up_proj_weight, up_proj_weight_scales,
        activation_scaler, weight_scaler, gemm_ref
    )
    
    # Element-wise multiplication (NOT quantized, remains in BF16)
    intermediate = gate_activated * up_output
    
    # FP8 down projection: [num_tokens, intermediate_size] @ [intermediate_size, hidden_size] -> [num_tokens, hidden_size]
    output = _fp8_linear(
        intermediate, down_proj_weight, down_proj_weight_scales,
        activation_scaler, weight_scaler, gemm_ref
    )
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.043493 ms |
| - | Scoring Baseline | 0.500000 | 1.034997 ms |
| - | Reference Implementation | 0.348255 | 1.891231 ms |
