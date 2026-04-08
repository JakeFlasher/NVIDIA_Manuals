## Description

FP8-quantized MoE router gate projection that computes expert routing logits. Projects hidden states to expert scores using blockwise FP8 quantization with BlockWise1x128 scaling for activations and BlockWise128x128 scaling for weights. The weight matrix is padded to 128 experts for BlockWise128x128 compatibility.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [M, hidden_size] | float8_e4m3fn |
| gate_weight | [padded_experts, hidden_size] | float8_e4m3fn |
| scale_hidden | [M, hidden_blocks] | float32 |
| scale_weight | [1, hidden_blocks] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| router_logits | [M, num_experts] | bfloat16 |

| # | hidden_size | num_experts | padded_experts | hidden_blocks | M | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3584 | 64 | 128 | 28 | 1024 | 0.0788 | 0.0006 |
| 2 | 3584 | 64 | 128 | 28 | 1408 | 0.1099 | 0.0008 |
| 3 | 3584 | 64 | 128 | 28 | 4096 | 0.1092 | 0.0020 |
| 4 | 3584 | 64 | 128 | 28 | 6144 | 0.0965 | 0.0030 |
| 5 | 3584 | 64 | 128 | 28 | 768 | 0.0830 | 0.0005 |
| 6 | 3584 | 64 | 128 | 28 | 3584 | 0.0826 | 0.0018 |
| 7 | 3584 | 64 | 128 | 28 | 896 | 0.0822 | 0.0005 |
| 8 | 3584 | 64 | 128 | 28 | 1280 | 0.0838 | 0.0007 |
| 9 | 3584 | 64 | 128 | 28 | 256 | 0.0827 | 0.0002 |
| 10 | 3584 | 64 | 128 | 28 | 4352 | 0.0815 | 0.0022 |
| 11 | 3584 | 64 | 128 | 28 | 3328 | 0.0823 | 0.0017 |
| 12 | 3584 | 64 | 128 | 28 | 3072 | 0.0844 | 0.0016 |
| 13 | 3584 | 64 | 128 | 28 | 4864 | 0.0883 | 0.0024 |
| 14 | 3584 | 64 | 128 | 28 | 2048 | 0.0820 | 0.0011 |
| 15 | 3584 | 64 | 128 | 28 | 8192 | 0.0976 | 0.0039 |
| 16 | 3584 | 64 | 128 | 28 | 5376 | 0.0912 | 0.0026 |

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
    """
    Generate FP8 quantized inputs with precomputed scales.
    
    Args:
        axes_and_scalars: Dictionary containing axis values
        device: Target device
    
    Returns:
        Dictionary of input tensors
    """
    M = axes_and_scalars["M"]
    hidden_size = 3584
    num_experts = 64
    padded_experts = 128
    
    # Generate random inputs in BF16
    hidden_states_bf16 = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
    gate_weight_bf16 = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Convert to FP32 for scale computation
    hidden_fp32 = hidden_states_bf16.to(torch.float32)
    weight_fp32 = gate_weight_bf16.to(torch.float32)
    
    # Pad weight to 128 experts for BlockWise128x128 compatibility
    weight_padded = torch.zeros(padded_experts, hidden_size, dtype=torch.float32, device=device)
    weight_padded[:num_experts, :] = weight_fp32
    
    # Transpose padded weight for blockwise scale computation: (hidden_size, padded_experts)
    weight_t = weight_padded.T  # (3584, 128)
    
    # Compute scales using unified interface
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    scale_hidden = activation_scaler.compute_scales(hidden_fp32)  # (M, hidden_size//128) = (M, 28)
    
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    scale_weight = weight_scaler.compute_scales(weight_t)  # (hidden_size//128, padded_experts//128) = (28, 1)
    
    # Apply scaling and quantize to FP8
    hidden_scaled = activation_scaler.apply_scaling(
        hidden_fp32, scale_hidden, inverse=False, clamp_to_fp8_range=True
    )
    weight_scaled = weight_scaler.apply_scaling(
        weight_t, scale_weight, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = hidden_scaled.to(torch.float8_e4m3fn)  # (M, hidden_size)
    # Transpose back to get (padded_experts, hidden_size)
    qw = weight_scaled.T.to(torch.float8_e4m3fn)  # (padded_experts, hidden_size)
    
    # Transpose scale_weight from (28, 1) to (1, 28) for the definition shape
    scale_weight_transposed = scale_weight.T.contiguous()  # (1, 28)
    
    return {
        "hidden_states": qx,
        "gate_weight": qw,
        "scale_hidden": scale_hidden,
        "scale_weight": scale_weight_transposed,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    scale_hidden: torch.Tensor,
    scale_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 router projection using blockwise scaling.
    
    Args:
        hidden_states: Quantized input [M, hidden_size] in FP8
        gate_weight: Quantized weight [padded_experts, hidden_size] in FP8
        scale_hidden: Blockwise scales for hidden_states [M, hidden_size//128]
        scale_weight: Blockwise scales for weight [1, hidden_size//128]
    
    Returns:
        router_logits: [M, num_experts] in BF16
    """
    num_experts = 64
    
    # scale_weight is (1, 28), need (padded_experts//128, hidden_size//128) = (1, 28) for CuBLAS
    # CuBLAS expects (N//128, K//128) where N=padded_experts=128, K=hidden_size=3584
    # So we need (1, 28) which is what we have
    scale_weight_cublas = scale_weight.contiguous()  # (1, 28)
    
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # FP8 GEMM: [M, K] @ [N, K]^T = [M, N]
    # mat_a: [M, hidden_size], mat_b: [padded_experts, hidden_size]
    router_logits_padded = gemm_ref.scaled_mm(
        mat_a=hidden_states,
        mat_b=gate_weight,
        scale_a=scale_hidden,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_weight_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Extract only the first num_experts columns
    router_logits = router_logits_padded[:, :num_experts]
    
    return router_logits

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"M": 1024},
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(out.shape)  # Expected: (1024, 64)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001246 ms |
| - | Scoring Baseline | 0.500000 | 0.088039 ms |
| - | Reference Implementation | 0.282129 | 0.229169 ms |
