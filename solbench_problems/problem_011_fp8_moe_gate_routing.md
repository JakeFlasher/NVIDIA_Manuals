## Description

FP8-quantized MoE gating mechanism with top-k expert selection using noaux_tc method. Computes expert scores via FP8 linear projection, applies sigmoid activation, adds correction bias, performs group-based top-k selection across 256 experts with 8 groups and 4 top groups. Returns top-8 expert indices and normalized weights per token.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| weight | [n_routed_experts, hidden_size] | bfloat16 |
| e_score_correction_bias | [n_routed_experts] | bfloat16 |
| scale_x | [num_tokens, hidden_blocks] | float32 |
| scale_w | [hidden_blocks, expert_blocks] | float32 |
| routed_scaling_factor | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| topk_idx | [num_tokens, num_experts_per_tok] | int64 |
| topk_weight | [num_tokens, num_experts_per_tok] | float32 |

| # | hidden_size | n_routed_experts | num_experts_per_tok | n_group | topk_group | hidden_blocks | expert_blocks | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 4352 | 0.7247 | 0.0047 |
| 2 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 5888 | 1.0482 | 0.0062 |
| 3 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 1280 | 0.4591 | 0.0017 |
| 4 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 768 | 0.3003 | 0.0012 |
| 5 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 4864 | 0.9370 | 0.0052 |
| 6 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 1536 | 0.4204 | 0.0020 |
| 7 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 2816 | 0.6170 | 0.0032 |
| 8 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 3328 | 0.6706 | 0.0037 |
| 9 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 2560 | 0.5687 | 0.0030 |
| 10 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 7168 | 1.1939 | 0.0075 |
| 11 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 4608 | 0.7523 | 0.0050 |
| 12 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 640 | 0.2858 | 0.0011 |
| 13 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 6656 | 1.0848 | 0.0070 |
| 14 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 256 | 0.2580 | 0.0007 |
| 15 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 8192 | 1.2872 | 0.0084 |
| 16 | 7168 | 256 | 8 | 8 | 4 | 56 | 2 | 1024 | 0.3725 | 0.0015 |

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

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with proper FP8 scales."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 7168
    n_routed_experts = 256
    
    # Generate random tensors
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n_routed_experts, hidden_size, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(n_routed_experts, dtype=torch.bfloat16, device=device) * 0.1
    
    # Compute FP8 scales
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    scale_x = activation_scaler.compute_scales(hidden_states_fp32)
    
    # Transpose weight to (K, N) for blockwise scaling
    weight_t = weight_fp32.T  # (7168, 256)
    scale_w = weight_scaler.compute_scales(weight_t)
    
    return {
        "hidden_states": hidden_states,
        "weight": weight,
        "e_score_correction_bias": e_score_correction_bias,
        "scale_x": scale_x,
        "scale_w": scale_w,
        "routed_scaling_factor": 2.5,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    FP8-quantized MoE gating with top-k expert selection.
    
    Steps:
    1. FP8 GEMM for gating scores: hidden_states @ weight.T
    2. Sigmoid activation on scores
    3. Add score correction bias for noaux_tc method
    4. Group-based top-k selection (8 groups, select top 4 groups)
    5. Final top-k expert selection (8 experts per token)
    6. Score normalization and scaling
    """
    # Constants
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_group = 8
    topk_group = 4
    
    bsz_seq_len = hidden_states.shape[0]
    
    # FP8 scaling infrastructure
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Step 1: FP8 GEMM for gating scores
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        hidden_states_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    
    # Transpose weight to (K, N) for blockwise scaling
    weight_t = weight_fp32.T  # (7168, 256)
    w_scaled = weight_scaler.apply_scaling(
        weight_t, scale_w, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # [bsz_seq_len, 7168]
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [256, 7168]
    
    # Transpose weight scales for CuBLAS format
    scale_w_cublas = scale_w.T.contiguous()  # [N//128, K//128]
    
    # FP8 GEMM: [bsz_seq_len, 7168] @ [256, 7168].T -> [bsz_seq_len, 256]
    logits = gemm_ref.scaled_mm(
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
    
    # Step 2: Sigmoid activation
    scores = torch.sigmoid(logits.to(torch.float32))
    
    # Step 3: Add score correction bias for noaux_tc method
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)
    
    # Step 4: Group-based top-k selection
    experts_per_group = n_routed_experts // n_group  # 32
    group_scores_reshaped = scores_for_choice.view(
        bsz_seq_len, n_group, experts_per_group
    )
    
    # Select top-2 experts per group and sum their scores
    group_scores = group_scores_reshaped.topk(2, dim=-1)[0].sum(dim=-1)
    
    # Select top-4 groups out of 8
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create group mask
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand mask to expert level
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz_seq_len, n_group, experts_per_group)
        .reshape(bsz_seq_len, n_routed_experts)
    )
    
    # Step 5: Mask out non-selected groups and perform final top-k
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(tmp_scores, k=num_experts_per_tok, dim=-1, sorted=False)
    
    # Step 6: Gather final weights and normalize
    topk_weight = scores.gather(1, topk_idx)
    
    # Normalize weights
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    
    # Apply routing scaling factor
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003056 ms |
| - | Scoring Baseline | 0.500000 | 0.606414 ms |
| - | Reference Implementation | 0.375950 | 1.005045 ms |
