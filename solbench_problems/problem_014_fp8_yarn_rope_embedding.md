## Description

FP8-quantized YaRN (Yet another RoPE extensioN method) rotary position embedding computation with dynamic frequency adjustment for extended context (163840 tokens). Computes cos/sin embeddings for rotary position encoding with YaRN interpolation and mscale factors.
| Name | Shape | Dtype |
| --- | --- | --- |
| position_ids | [seq_len] | int64 |
| mscale | scalar | float32 |
| mscale_all_dim | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| cos_emb | [seq_len, qk_rope_head_dim] | bfloat16 |
| sin_emb | [seq_len, qk_rope_head_dim] | bfloat16 |

| # | qk_rope_head_dim | max_position_embeddings | original_max_position_embeddings | rope_theta | scaling_factor | beta_fast | beta_slow | half_dim | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 98304 | 0.0909 | 0.0017 |
| 2 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 12288 | 0.0324 | 0.0002 |
| 3 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 2 | 0.0235 | 0.0000 |
| 4 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 1024 | 0.0279 | 0.0000 |
| 5 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 896 | 0.0279 | 0.0000 |
| 6 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 2048 | 0.0284 | 0.0000 |
| 7 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 16384 | 0.0333 | 0.0003 |
| 8 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 3072 | 0.0280 | 0.0001 |
| 9 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 4 | 0.0230 | 0.0000 |
| 10 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 24576 | 0.0398 | 0.0004 |
| 11 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 131072 | 0.1272 | 0.0022 |
| 12 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 32 | 0.0269 | 0.0000 |
| 13 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 8 | 0.0268 | 0.0000 |
| 14 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 64 | 0.0266 | 0.0000 |
| 15 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 65536 | 0.0670 | 0.0011 |
| 16 | 64 | 163840 | 4096 | 10000 | 40 | 32 | 1 | 32 | 1 | 0.0267 | 0.0000 |

```python
import torch
import math
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

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Find dimension for correction based on number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dimension range bounds based on rotations."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

def yarn_get_mscale(scale=1, mscale=1):
    """Compute mscale factor for YaRN."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def yarn_linear_ramp_mask(min_val, max_val, dim, device):
    """Create linear ramp mask for frequency interpolation."""
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for YaRN RoPE embedding computation."""
    seq_len = axes_and_scalars["seq_len"]
    
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device)
    mscale = 1.0
    mscale_all_dim = 1.0
    
    return {
        "position_ids": position_ids,
        "mscale": mscale,
        "mscale_all_dim": mscale_all_dim,
    }

@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    mscale: float,
    mscale_all_dim: float,
):
    """
    Compute FP8-quantized YaRN rotary position embeddings.
    
    This implements YaRN (Yet another RoPE extensioN method) with FP8 quantization
    for memory bandwidth reduction. The computation involves:
    1. Computing base inverse frequencies
    2. Applying YaRN interpolation with correction ranges
    3. FP8 quantization of frequency computations
    4. Computing cos/sin embeddings with mscale
    
    Args:
        position_ids: Position indices (seq_len,)
        mscale: Mscale factor for YaRN
        mscale_all_dim: Mscale all dimension factor
    
    Returns:
        Tuple of (cos_emb, sin_emb) tensors for rotary embedding application
    """
    device = position_ids.device
    seq_len = position_ids.shape[0]
    
    # Constants from definition
    dim = 64  # qk_rope_head_dim
    base = 10000  # rope_theta
    scaling_factor = 40
    original_max_position_embeddings = 4096
    beta_fast = 32
    beta_slow = 1
    
    # FP8 scaler for frequency computations
    freq_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    
    # Compute base frequencies
    freq_extra = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    freq_inter = 1.0 / (
        scaling_factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    
    # Find correction range for YaRN interpolation
    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        base,
        original_max_position_embeddings,
    )
    
    # Create interpolation mask
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2, device)
    
    # Interpolate frequencies using YaRN method
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    
    # Create position tensor from position_ids
    t = position_ids.to(torch.float32)
    
    # Compute frequencies via outer product
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim//2)
    
    # Pad to multiple of 128 for blockwise FP8 scaling
    pad_size = (128 - (freqs.shape[1] % 128)) % 128
    if pad_size > 0:
        freqs_padded = torch.nn.functional.pad(freqs, (0, pad_size), value=0.0)
    else:
        freqs_padded = freqs
    
    # FP8 quantization of frequencies
    scales = freq_scaler.compute_scales(freqs_padded)
    freqs_scaled = freq_scaler.apply_scaling(
        freqs_padded, scales, inverse=False, clamp_to_fp8_range=True
    )
    freqs_fp8 = freqs_scaled.to(torch.float8_e4m3fn)
    
    # Dequantize for cos/sin computation
    freqs_dequant = freqs_fp8.to(torch.float32)
    freqs_dequant = freq_scaler.apply_scaling(
        freqs_dequant, scales, inverse=True
    )
    
    # Remove padding
    freqs_final = freqs_dequant[:, :dim // 2]
    
    # Compute mscale factor
    _mscale = float(
        yarn_get_mscale(scaling_factor, mscale)
        / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )
    
    # Concatenate for full embedding dimension
    emb = torch.cat((freqs_final, freqs_final), dim=-1)  # (seq_len, dim)
    
    # Compute cos/sin with mscale
    cos_emb = (emb.cos() * _mscale).to(torch.bfloat16)
    sin_emb = (emb.sin() * _mscale).to(torch.bfloat16)
    
    return cos_emb, sin_emb

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"seq_len": 1024},
        device=torch.device("cuda:0"),
    )
    cos_emb, sin_emb = run(**inputs)
    print(f"cos_emb shape: {cos_emb.shape}")
    print(f"sin_emb shape: {sin_emb.shape}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000014 ms |
| - | Scoring Baseline | 0.500000 | 0.035257 ms |
| - | Reference Implementation | 0.100315 | 0.327514 ms |
