## Description

FP8-quantized fused gate and up projection for standard MLP layers. Combines gate_proj and up_proj linear projections with SiLU activation: output = silu(gate_proj(x)) * up_proj(x). Uses BlockWise1x128 scaling for input activations and BlockWise128x128 scaling for weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [M, hidden_size] | float8_e4m3fn |
| scale_x | [M, hidden_blocks] | float32 |
| gate_proj_weight | [intermediate_size, hidden_size] | float8_e4m3fn |
| scale_gate | [intermediate_blocks, hidden_blocks] | float32 |
| up_proj_weight | [intermediate_size, hidden_size] | float8_e4m3fn |
| scale_up | [intermediate_blocks, hidden_blocks] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [M, intermediate_size] | bfloat16 |

| # | hidden_size | intermediate_size | hidden_blocks | intermediate_blocks | M | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3584 | 18944 | 28 | 148 | 3584 | 0.7935 | 0.2687 |
| 2 | 3584 | 18944 | 28 | 148 | 1152 | 0.3952 | 0.0864 |
| 3 | 3584 | 18944 | 28 | 148 | 8704 | 1.8466 | 0.6525 |
| 4 | 3584 | 18944 | 28 | 148 | 512 | 0.2524 | 0.0384 |
| 5 | 3584 | 18944 | 28 | 148 | 2816 | 0.6633 | 0.2111 |
| 6 | 3584 | 18944 | 28 | 148 | 2304 | 0.5996 | 0.1727 |
| 7 | 3584 | 18944 | 28 | 148 | 5632 | 1.1908 | 0.4222 |
| 8 | 3584 | 18944 | 28 | 148 | 3840 | 1.0569 | 0.2879 |
| 9 | 3584 | 18944 | 28 | 148 | 1792 | 0.4850 | 0.1343 |
| 10 | 3584 | 18944 | 28 | 148 | 6144 | 1.2857 | 0.4606 |
| 11 | 3584 | 18944 | 28 | 148 | 384 | 0.2324 | 0.0367 |
| 12 | 3584 | 18944 | 28 | 148 | 3072 | 0.7171 | 0.2303 |
| 13 | 3584 | 18944 | 28 | 148 | 768 | 0.3022 | 0.0576 |
| 14 | 3584 | 18944 | 28 | 148 | 2048 | 0.5380 | 0.1535 |
| 15 | 3584 | 18944 | 28 | 148 | 5120 | 1.0754 | 0.3838 |
| 16 | 3584 | 18944 | 28 | 148 | 1280 | 0.4005 | 0.0960 |

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
    """Generate FP8 quantized inputs with proper scales."""
    M = axes_and_scalars["M"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    
    # Generate random FP32 tensors - scale weights by 1/sqrt(fan_in) to keep output magnitudes reasonable
    x_fp32 = torch.randn(M, hidden_size, dtype=torch.float32, device=device)
    gate_weight_fp32 = torch.randn(intermediate_size, hidden_size, dtype=torch.float32, device=device) * (hidden_size ** -0.5)
    up_weight_fp32 = torch.randn(intermediate_size, hidden_size, dtype=torch.float32, device=device) * (hidden_size ** -0.5)
    
    # Create scalers
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    # Compute scales for input (BlockWise1x128)
    scale_x = activation_scaler.compute_scales(x_fp32)
    
    # Compute scales for weights (BlockWise128x128)
    # Transpose weights to (K, N) for blockwise scale computation
    gate_weight_t = gate_weight_fp32.T  # (hidden_size, intermediate_size)
    up_weight_t = up_weight_fp32.T  # (hidden_size, intermediate_size)
    
    scales_gate = weight_scaler.compute_scales(gate_weight_t)
    scales_up = weight_scaler.compute_scales(up_weight_t)
    
    # Apply scaling and quantize input
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    qx = x_scaled.to(torch.float8_e4m3fn)
    
    # Apply scaling and quantize gate weight
    gate_scaled = weight_scaler.apply_scaling(
        gate_weight_t, scales_gate, inverse=False, clamp_to_fp8_range=True
    )
    qw_gate = gate_scaled.T.to(torch.float8_e4m3fn)  # (N, K)
    
    # Apply scaling and quantize up weight
    up_scaled = weight_scaler.apply_scaling(
        up_weight_t, scales_up, inverse=False, clamp_to_fp8_range=True
    )
    qw_up = up_scaled.T.to(torch.float8_e4m3fn)  # (N, K)
    
    # Transpose weight scales for CuBLAS format: (K//128, N//128) -> (N//128, K//128)
    scale_gate_cublas = scales_gate.T.contiguous()
    scale_up_cublas = scales_up.T.contiguous()
    
    return {
        "x": qx,
        "scale_x": scale_x,
        "gate_proj_weight": qw_gate,
        "scale_gate": scale_gate_cublas,
        "up_proj_weight": qw_up,
        "scale_up": scale_up_cublas,
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    scale_x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    scale_gate: torch.Tensor,
    up_proj_weight: torch.Tensor,
    scale_up: torch.Tensor,
):
    """
    FP8 fused gate-up projection with SiLU activation.
    
    Computes: output = silu(gate_proj(x)) * up_proj(x)
    
    Args:
        x: Quantized input [M, hidden_size] in FP8
        scale_x: Input scales [M, hidden_size//128] for BlockWise1x128
        gate_proj_weight: Quantized gate weight [intermediate_size, hidden_size] in FP8
        scale_gate: Gate weight scales [intermediate_size//128, hidden_size//128]
        up_proj_weight: Quantized up weight [intermediate_size, hidden_size] in FP8
        scale_up: Up weight scales [intermediate_size//128, hidden_size//128]
    
    Returns:
        Output tensor [M, intermediate_size] in bfloat16
    """
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # FP8 GEMM for gate projection
    gate_output = gemm_ref.scaled_mm(
        mat_a=x,
        mat_b=gate_proj_weight,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_gate,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # FP8 GEMM for up projection
    up_output = gemm_ref.scaled_mm(
        mat_a=x,
        mat_b=up_proj_weight,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_up,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Apply SiLU activation to gate output
    gate_activated = F.silu(gate_output)
    
    # Element-wise multiplication
    output = gate_activated * up_output
    
    return output

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"M": 1024, "hidden_size": 3584, "intermediate_size": 18944},
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.168914 ms |
| - | Scoring Baseline | 0.500000 | 0.624306 ms |
| - | Reference Implementation | 0.032857 | 13.900113 ms |
