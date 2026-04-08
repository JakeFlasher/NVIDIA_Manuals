## Description

FP8-quantized shared expert MLP with SwiGLU activation. Processes ALL tokens in MoE layers with blockwise FP8 quantization (1x128 for activations, 128x128 for weights). Architecture: x -> [gate_proj, up_proj] -> silu(gate) * up -> down_proj -> output.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | hidden_blocks | intermediate_blocks | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 2048 | 56 | 16 | 2816 | 7.2903 | 0.0685 |
| 2 | 7168 | 2048 | 56 | 16 | 3840 | 9.3535 | 0.0934 |
| 3 | 7168 | 2048 | 56 | 16 | 640 | 2.9313 | 0.0156 |
| 4 | 7168 | 2048 | 56 | 16 | 2688 | 6.8762 | 0.0654 |
| 5 | 7168 | 2048 | 56 | 16 | 3328 | 8.1233 | 0.0809 |
| 6 | 7168 | 2048 | 56 | 16 | 4096 | 10.2869 | 0.0996 |
| 7 | 7168 | 2048 | 56 | 16 | 2432 | 6.8267 | 0.0591 |
| 8 | 7168 | 2048 | 56 | 16 | 256 | 1.8386 | 0.0122 |
| 9 | 7168 | 2048 | 56 | 16 | 1408 | 4.4448 | 0.0342 |
| 10 | 7168 | 2048 | 56 | 16 | 768 | 3.1101 | 0.0187 |
| 11 | 7168 | 2048 | 56 | 16 | 2560 | 6.8495 | 0.0622 |
| 12 | 7168 | 2048 | 56 | 16 | 3968 | 9.3727 | 0.0965 |
| 13 | 7168 | 2048 | 56 | 16 | 512 | 2.3525 | 0.0129 |
| 14 | 7168 | 2048 | 56 | 16 | 2944 | 8.0269 | 0.0716 |
| 15 | 7168 | 2048 | 56 | 16 | 2304 | 5.9015 | 0.0560 |
| 16 | 7168 | 2048 | 56 | 16 | 896 | 3.3134 | 0.0218 |

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
    """Generate inputs with scaled weights to keep outputs in reasonable range."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 7168
    intermediate_size = 2048

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    gate_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    up_proj_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    down_proj_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * (intermediate_size ** -0.5)

    return {
        "hidden_states": hidden_states,
        "gate_proj_weight": gate_proj_weight,
        "up_proj_weight": up_proj_weight,
        "down_proj_weight": down_proj_weight,
    }

def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation_scaler: BlockwiseScaler,
    weight_scaler: BlockwiseScaler,
    gemm_ref: CuBLASRefBlockwiseGemm,
) -> torch.Tensor:
    """
    FP8 linear layer with blockwise scaling.
    
    Args:
        x: Input tensor (M, K) in BF16
        weight: Weight tensor (N, K) in BF16
    
    Returns:
        Output tensor (M, N) in BF16
    """
    # Convert to FP32 for scaling
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Compute activation scales dynamically
    scale_x = activation_scaler.compute_scales(x_fp32)
    
    # Transpose weight for blockwise scaling (weight is N, K, need K, N)
    weight_t = weight_fp32.T  # (K, N)
    weight_scales = weight_scaler.compute_scales(weight_t)
    
    # Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    
    weight_scaled = weight_scaler.apply_scaling(
        weight_t, weight_scales, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # (M, K)
    qw = weight_scaled.T.to(torch.float8_e4m3fn)  # (N, K)
    
    # Transpose weight scales for cuBLAS format
    scale_w_cublas = weight_scales.T.contiguous()  # (N//128, K//128)
    
    # FP8 GEMM using cuBLAS reference
    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    return output

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 Shared Expert MLP with SwiGLU activation.
    
    Architecture: x -> [gate_proj, up_proj] -> silu(gate) * up -> down_proj -> output
    
    Uses blockwise FP8 quantization:
    - Activations: BlockWise1x128 (per-tensor in M, 128-sized blocks in K)
    - Weights: BlockWise128x128 (128x128 blockwise)
    
    Args:
        hidden_states: Input tensor (num_tokens, hidden_size)
        gate_proj_weight: Gate projection weight (intermediate_size, hidden_size)
        up_proj_weight: Up projection weight (intermediate_size, hidden_size)
        down_proj_weight: Down projection weight (hidden_size, intermediate_size)
    
    Returns:
        Output tensor (num_tokens, hidden_size)
    """
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # FP8 gate projection: (num_tokens, hidden_size) @ (hidden_size, intermediate_size) -> (num_tokens, intermediate_size)
    gate = _fp8_linear(
        hidden_states,
        gate_proj_weight,
        activation_scaler,
        weight_scaler,
        gemm_ref,
    )
    
    # FP8 up projection: (num_tokens, hidden_size) @ (hidden_size, intermediate_size) -> (num_tokens, intermediate_size)
    up = _fp8_linear(
        hidden_states,
        up_proj_weight,
        activation_scaler,
        weight_scaler,
        gemm_ref,
    )
    
    # SwiGLU activation (NOT quantized - remains in BF16)
    # silu(x) = x * sigmoid(x)
    intermediate = F.silu(gate) * up
    
    # FP8 down projection: (num_tokens, intermediate_size) @ (intermediate_size, hidden_size) -> (num_tokens, hidden_size)
    output = _fp8_linear(
        intermediate,
        down_proj_weight,
        activation_scaler,
        weight_scaler,
        gemm_ref,
    )
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.043493 ms |
| - | Scoring Baseline | 0.500000 | 5.365907 ms |
| - | Reference Implementation | 0.471105 | 6.020466 ms |
