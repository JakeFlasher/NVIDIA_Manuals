## Description

FP8-quantized Multi-head Latent Attention (MLA) KV compression projection with low-rank bottleneck. Compresses hidden_size (7168) to kv_lora_rank + qk_rope_head_dim (576), applies RMSNorm on compressed representation, then expands to num_heads * (qk_nope_head_dim + v_head_dim) (24576). Uses BlockWise1x128 scaling for activations and BlockWise128x128 scaling for weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| kv_a_proj_weight | [640, hidden_size] | bfloat16 |
| kv_a_layernorm_weight | [kv_lora_rank] | bfloat16 |
| kv_b_proj_weight | [kv_b_output_dim, kv_lora_rank] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| kv_expanded | [batch_size, seq_len, num_heads, 256] | bfloat16 |
| k_pe | [batch_size, seq_len, 1, qk_rope_head_dim] | bfloat16 |

| # | hidden_size | num_heads | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | kv_a_output_dim | kv_b_output_dim | hidden_blocks | kv_a_out_blocks | kv_lora_blocks | kv_b_out_blocks | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 64 | 128 | 1.7286 | 0.0966 |
| 2 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 4 | 512 | 0.7189 | 0.0242 |
| 3 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 64 | 256 | 3.0481 | 0.1933 |
| 4 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 4 | 256 | 0.6027 | 0.0121 |
| 5 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 8 | 256 | 0.7198 | 0.0242 |
| 6 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 32 | 128 | 1.0679 | 0.0483 |
| 7 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 32 | 256 | 1.7302 | 0.0966 |
| 8 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 32 | 512 | 3.0503 | 0.1933 |
| 9 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 2 | 4096 | 1.7295 | 0.0966 |
| 10 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 16 | 128 | 0.7203 | 0.0242 |
| 11 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 1 | 256 | 0.5423 | 0.0056 |
| 12 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 2 | 2048 | 1.0673 | 0.0483 |
| 13 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 2 | 512 | 0.6033 | 0.0121 |
| 14 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 4 | 1024 | 1.0686 | 0.0483 |
| 15 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 8 | 512 | 1.0671 | 0.0483 |
| 16 | 7168 | 128 | 512 | 64 | 128 | 128 | 576 | 32768 | 56 | 4 | 4 | 256 | 1 | 1024 | 0.5982 | 0.0121 |

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

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 7168
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    
    kv_a_output_dim_padded = 640  # Padded from 576 to 640 for FP8 alignment
    kv_b_output_dim = num_heads * (qk_nope_head_dim + v_head_dim)  # 24576
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    kv_a_proj_weight = torch.randn(kv_a_output_dim_padded, hidden_size, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_weight = torch.ones(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_weight = torch.randn(kv_b_output_dim, kv_lora_rank, dtype=torch.bfloat16, device=device)
    
    return {
        "hidden_states": hidden_states,
        "kv_a_proj_weight": kv_a_proj_weight,
        "kv_a_layernorm_weight": kv_a_layernorm_weight,
        "kv_b_proj_weight": kv_b_proj_weight,
        "rms_norm_eps": 1e-6,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """
    FP8 MLA KV compression projection.
    
    1. FP8 compression: hidden_size (7168) -> kv_lora_rank + qk_rope_head_dim (576, padded to 640)
    2. Split and RMSNorm on compressed representation (first 512 dims)
    3. FP8 expansion: kv_lora_rank (512) -> num_heads * (qk_nope_head_dim + v_head_dim) (24576)
    """
    # Constants
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    
    bsz, q_len, hidden_size = hidden_states.shape
    
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Reshape for linear: (bsz, seq_len, hidden_size) -> (bsz * seq_len, hidden_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    M = hidden_flat.shape[0]
    
    # ===== Step 1: FP8 Compression Projection =====
    # hidden_flat: (M, 7168), kv_a_proj_weight: (640, 7168)
    x_fp32 = hidden_flat.to(torch.float32)
    w_a_fp32 = kv_a_proj_weight.to(torch.float32)
    
    # Compute scales
    scale_x_a = activation_scaler.compute_scales(x_fp32)
    w_a_fp32_t = w_a_fp32.T  # (7168, 640)
    scale_w_a = weight_scaler.compute_scales(w_a_fp32_t)
    
    # Apply scaling and quantize
    x_scaled_a = activation_scaler.apply_scaling(x_fp32, scale_x_a, inverse=False, clamp_to_fp8_range=True)
    w_scaled_a = weight_scaler.apply_scaling(w_a_fp32_t, scale_w_a, inverse=False, clamp_to_fp8_range=True)
    
    qx_a = x_scaled_a.to(torch.float8_e4m3fn)
    qw_a = w_scaled_a.T.to(torch.float8_e4m3fn)  # (640, 7168)
    
    # FP8 GEMM
    scale_w_a_cublas = scale_w_a.T.contiguous()
    compressed_kv_with_rope = gemm_ref.scaled_mm(
        mat_a=qx_a,
        mat_b=qw_a,
        scale_a=scale_x_a,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_a_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )  # (M, 640)
    
    # ===== Step 2: Split and RMSNorm =====
    # Split: first 512 dims for compressed_kv, next 64 for k_pe (ignore padding)
    compressed_kv = compressed_kv_with_rope[:, :kv_lora_rank]  # (M, 512)
    k_pe = compressed_kv_with_rope[:, kv_lora_rank:kv_lora_rank + qk_rope_head_dim]  # (M, 64)
    
    # RMSNorm on compressed_kv
    compressed_kv_fp32 = compressed_kv.to(torch.float32)
    variance = compressed_kv_fp32.pow(2).mean(-1, keepdim=True)
    compressed_kv_norm = compressed_kv_fp32 * torch.rsqrt(variance + rms_norm_eps)
    compressed_kv_norm = (kv_a_layernorm_weight * compressed_kv_norm.to(kv_a_layernorm_weight.dtype))
    
    # ===== Step 3: FP8 Expansion Projection =====
    # compressed_kv_norm: (M, 512), kv_b_proj_weight: (24576, 512)
    x_b_fp32 = compressed_kv_norm.to(torch.float32)
    w_b_fp32 = kv_b_proj_weight.to(torch.float32)
    
    # Compute scales
    scale_x_b = activation_scaler.compute_scales(x_b_fp32)
    w_b_fp32_t = w_b_fp32.T  # (512, 24576)
    scale_w_b = weight_scaler.compute_scales(w_b_fp32_t)
    
    # Apply scaling and quantize
    x_scaled_b = activation_scaler.apply_scaling(x_b_fp32, scale_x_b, inverse=False, clamp_to_fp8_range=True)
    w_scaled_b = weight_scaler.apply_scaling(w_b_fp32_t, scale_w_b, inverse=False, clamp_to_fp8_range=True)
    
    qx_b = x_scaled_b.to(torch.float8_e4m3fn)
    qw_b = w_scaled_b.T.to(torch.float8_e4m3fn)  # (24576, 512)
    
    # FP8 GEMM
    scale_w_b_cublas = scale_w_b.T.contiguous()
    kv_expanded_flat = gemm_ref.scaled_mm(
        mat_a=qx_b,
        mat_b=qw_b,
        scale_a=scale_x_b,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_b_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )  # (M, 24576)
    
    # ===== Reshape outputs =====
    kv_expanded = kv_expanded_flat.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim)
    
    return kv_expanded, k_pe

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"batch_size": 2, "seq_len": 128},
        device=torch.device("cuda:0"),
    )
    kv_expanded, k_pe = run(**inputs)
    print(f"kv_expanded shape: {kv_expanded.shape}")
    print(f"k_pe shape: {k_pe.shape}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.038718 ms |
| - | Scoring Baseline | 0.500000 | 1.065229 ms |
| - | Reference Implementation | 0.371765 | 1.771065 ms |
