## Description

FP8-quantized output projection in Mamba2 mixer that projects from intermediate size (2688) back to hidden dimension (2688). Uses BlockWise1x128 scaling for activations and BlockWise128x128 scaling for weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [M, K] | float8_e4m3fn |
| w | [N, K] | float8_e4m3fn |
| scale_x | [M, K_blocks] | float32 |
| scale_w | [N_blocks, K_blocks] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [M, N] | bfloat16 |

| # | K | N | BLOCK | K_blocks | N_blocks | M | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2688 | 2688 | 128 | 21 | 21 | 7680 | 0.3254 | 0.0306 |
| 2 | 2688 | 2688 | 128 | 21 | 21 | 6656 | 0.1775 | 0.0266 |
| 3 | 2688 | 2688 | 128 | 21 | 21 | 256 | 0.1013 | 0.0021 |
| 4 | 2688 | 2688 | 128 | 21 | 21 | 2560 | 0.1225 | 0.0102 |
| 5 | 2688 | 2688 | 128 | 21 | 21 | 1024 | 0.1048 | 0.0041 |
| 6 | 2688 | 2688 | 128 | 21 | 21 | 7168 | 0.1825 | 0.0286 |
| 7 | 2688 | 2688 | 128 | 21 | 21 | 4608 | 0.1514 | 0.0184 |
| 8 | 2688 | 2688 | 128 | 21 | 21 | 512 | 0.0997 | 0.0022 |
| 9 | 2688 | 2688 | 128 | 21 | 21 | 2816 | 0.1328 | 0.0112 |
| 10 | 2688 | 2688 | 128 | 21 | 21 | 4352 | 0.1485 | 0.0174 |
| 11 | 2688 | 2688 | 128 | 21 | 21 | 1792 | 0.1120 | 0.0071 |
| 12 | 2688 | 2688 | 128 | 21 | 21 | 4096 | 0.1479 | 0.0163 |
| 13 | 2688 | 2688 | 128 | 21 | 21 | 1152 | 0.1054 | 0.0046 |
| 14 | 2688 | 2688 | 128 | 21 | 21 | 5632 | 0.1698 | 0.0225 |
| 15 | 2688 | 2688 | 128 | 21 | 21 | 640 | 0.1014 | 0.0026 |
| 16 | 2688 | 2688 | 128 | 21 | 21 | 128 | 0.0985 | 0.0020 |

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
    """Generate inputs with proper FP8 quantization and scaling."""
    M = axes_and_scalars["M"]
    K = 2688
    N = 2688
    BLOCK = 128
    
    # Generate random BF16 tensors
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    
    # Convert to FP32 for scale computation
    x_fp32 = x_bf16.to(torch.float32)
    w_fp32 = w_bf16.to(torch.float32)
    
    # Create scalers - use BlockWise1x128 for activations (supported by CuBLAS)
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    # Compute scales
    scale_x = activation_scaler.compute_scales(x_fp32)  # [M, K//128]
    
    # Transpose weight for blockwise scaling: [N, K] -> [K, N]
    w_fp32_t = w_fp32.T.contiguous()
    scales_w = weight_scaler.compute_scales(w_fp32_t)  # [K//128, N//128]
    
    # Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    w_scaled = weight_scaler.apply_scaling(
        w_fp32_t, scales_w, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # [M, K]
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [N, K] - transpose back
    
    # Transpose weight scales for CuBLAS format: [K//128, N//128] -> [N//128, K//128]
    scale_w_cublas = scales_w.T.contiguous()
    
    return {
        "x": qx,
        "w": qw,
        "scale_x": scale_x,
        "scale_w": scale_w_cublas,
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
):
    """
    FP8 output projection GEMM with BlockWise1x128 activation scaling and BlockWise128x128 weight scaling.
    
    Args:
        x: Quantized input tensor [M, K] in FP8
        w: Quantized weight tensor [N, K] in FP8
        scale_x: BlockWise1x128 scales for input [M, K//128]
        scale_w: BlockWise128x128 scales for weights [N//128, K//128]
    
    Returns:
        Output tensor [M, N] in BF16
    """
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    output = gemm_ref.scaled_mm(
        mat_a=x,
        mat_b=w,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    return output

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"M": 1024},
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.008648 ms |
| - | Scoring Baseline | 0.500000 | 0.134932 ms |
| - | Reference Implementation | 0.396450 | 0.199712 ms |
