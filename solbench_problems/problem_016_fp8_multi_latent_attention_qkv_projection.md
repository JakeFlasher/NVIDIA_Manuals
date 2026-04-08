## Description

FP8-quantized Multi-Latent Attention (MLA) QKV projection with low-rank compression. Implements DeepSeek-V3's novel MLA mechanism with Q path (q_a_proj -> RMSNorm -> q_b_proj) and KV path (kv_a_proj_with_mqa -> split -> RMSNorm -> kv_b_proj). All linear layers use FP8 quantization with blockwise scaling. The kv_a_proj output dimension is padded to 640 (multiple of 128) for FP8 compatibility.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_a_proj_weight | [q_lora_rank, hidden_size] | bfloat16 |
| q_a_layernorm_weight | [q_lora_rank] | bfloat16 |
| q_b_proj_weight | [q_b_out_dim, q_lora_rank] | bfloat16 |
| kv_a_proj_weight | [kv_a_out_dim_padded, hidden_size] | bfloat16 |
| kv_a_layernorm_weight | [kv_lora_rank] | bfloat16 |
| kv_b_proj_weight | [kv_b_out_dim, kv_lora_rank] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| q_nope | [batch_size, seq_len, num_heads, qk_nope_head_dim] | bfloat16 |
| q_pe | [batch_size, seq_len, num_heads, qk_rope_head_dim] | bfloat16 |
| k_nope | [batch_size, seq_len, num_heads, qk_nope_head_dim] | bfloat16 |
| k_pe | [batch_size, seq_len, 1, qk_rope_head_dim] | bfloat16 |
| value_states | [batch_size, seq_len, num_heads, v_head_dim] | bfloat16 |

| # | hidden_size | num_heads | q_lora_rank | kv_lora_rank | qk_rope_head_dim | qk_nope_head_dim | v_head_dim | q_head_dim | kv_a_out_dim_padded | block_size | q_b_out_dim | kv_b_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 2 | 2048 | 2.4187 | 0.1586 |
| 2 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 1 | 512 | 0.8472 | 0.0198 |
| 3 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 4 | 128 | 0.9005 | 0.0198 |
| 4 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 8 | 2048 | 8.0857 | 0.6343 |
| 5 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 16 | 512 | 4.3148 | 0.3172 |
| 6 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 8 | 256 | 1.4753 | 0.0793 |
| 7 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 32 | 128 | 2.4198 | 0.1586 |
| 8 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 1 | 4096 | 2.4224 | 0.1586 |
| 9 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 2 | 512 | 1.0447 | 0.0396 |
| 10 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 1 | 256 | 0.7709 | 0.0184 |
| 11 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 1 | 128 | 0.7549 | 0.0183 |
| 12 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 32 | 512 | 8.0769 | 0.6343 |
| 13 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 16 | 1024 | 8.0706 | 0.6343 |
| 14 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 8 | 128 | 1.0494 | 0.0396 |
| 15 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 1 | 1024 | 1.0396 | 0.0396 |
| 16 | 7168 | 128 | 1536 | 512 | 64 | 128 | 128 | 192 | 640 | 128 | 24576 | 32768 | 8 | 1024 | 4.3177 | 0.3172 |

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

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm implementation."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)

def _fp8_linear(x: torch.Tensor, weight: torch.Tensor,
                activation_scaler: BlockwiseScaler,
                weight_scaler: BlockwiseScaler,
                gemm_ref: CuBLASRefBlockwiseGemm) -> torch.Tensor:
    """
    FP8 linear layer with blockwise scaling.
    
    Args:
        x: Input tensor [M, K] in bfloat16
        weight: Weight tensor [N, K] in bfloat16
    
    Returns:
        Output tensor [M, N] in bfloat16
    """
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w
    
    # Convert to FP32 for scaling
    x_fp32 = x.to(torch.float32)
    w_fp32 = weight.T.to(torch.float32)  # [K, N]
    
    # Compute scales
    scale_x = activation_scaler.compute_scales(x_fp32)
    scale_w = weight_scaler.compute_scales(w_fp32)
    
    # Apply scaling and quantize
    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    w_scaled = weight_scaler.apply_scaling(
        w_fp32, scale_w, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # [M, K]
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [N, K]
    
    # Transpose weight scales for CuBLAS format
    scale_w_cublas = scale_w.T.contiguous()  # [N//128, K//128]
    
    # FP8 GEMM
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
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """
    FP8-quantized Multi-Latent Attention QKV projection.
    
    Args:
        hidden_states: [B, L, 7168]
        q_a_proj_weight: [1536, 7168]
        q_a_layernorm_weight: [1536]
        q_b_proj_weight: [24576, 1536]
        kv_a_proj_weight: [640, 7168] (padded from 576 to 640 for FP8)
        kv_a_layernorm_weight: [512]
        kv_b_proj_weight: [32768, 512]
        rms_norm_eps: float
    
    Returns:
        q_nope: [B, L, 128, 128]
        q_pe: [B, L, 128, 64]
        k_nope: [B, L, 128, 128]
        k_pe: [B, L, 1, 64]
        value_states: [B, L, 128, 128]
    """
    # Constants
    num_heads = 128
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    q_head_dim = 192
    
    bsz, q_len, hidden_size = hidden_states.shape
    
    # Initialize scalers and GEMM reference
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Reshape for linear layers: [B, L, H] -> [B*L, H]
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    
    # === Q Path with Low-Rank Factorization ===
    # q_a_proj: [B*L, 7168] -> [B*L, 1536]
    q_a = _fp8_linear(hidden_flat, q_a_proj_weight, activation_scaler, weight_scaler, gemm_ref)
    
    # RMSNorm
    q_a = q_a.reshape(bsz, q_len, q_lora_rank)
    q_a_norm = _rms_norm(q_a, q_a_layernorm_weight, rms_norm_eps)
    q_a_norm_flat = q_a_norm.reshape(-1, q_lora_rank)
    
    # q_b_proj: [B*L, 1536] -> [B*L, 24576]
    q = _fp8_linear(q_a_norm_flat, q_b_proj_weight, activation_scaler, weight_scaler, gemm_ref)
    
    # Reshape and split Q: [B*L, 24576] -> [B, L, 128, 192] -> q_nope[128] + q_pe[64]
    q = q.view(bsz, q_len, num_heads, q_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    
    # === KV Path with Mandatory Low-Rank Compression ===
    # kv_a_proj_with_mqa: [B*L, 7168] -> [B*L, 640] (padded output)
    compressed_kv_with_pe_padded = _fp8_linear(hidden_flat, kv_a_proj_weight, activation_scaler, weight_scaler, gemm_ref)
    
    # Extract only the first 576 elements (512 + 64) from the padded output
    compressed_kv_with_pe = compressed_kv_with_pe_padded[:, :kv_lora_rank + qk_rope_head_dim]
    
    # Split into compressed_kv [512] and k_pe [64]
    compressed_kv_with_pe = compressed_kv_with_pe.reshape(bsz, q_len, kv_lora_rank + qk_rope_head_dim)
    compressed_kv, k_pe = torch.split(compressed_kv_with_pe, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    
    # Reshape k_pe: [B, L, 64] -> [B, L, 1, 64]
    k_pe = k_pe.unsqueeze(2)
    
    # RMSNorm on compressed KV
    compressed_kv_norm = _rms_norm(compressed_kv, kv_a_layernorm_weight, rms_norm_eps)
    compressed_kv_norm_flat = compressed_kv_norm.reshape(-1, kv_lora_rank)
    
    # kv_b_proj: [B*L, 512] -> [B*L, 32768]
    kv = _fp8_linear(compressed_kv_norm_flat, kv_b_proj_weight, activation_scaler, weight_scaler, gemm_ref)
    
    # Reshape and split KV: [B*L, 32768] -> [B, L, 128, 256] -> k_nope[128] + v[128]
    kv = kv.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    
    return q_nope, q_pe, k_nope, k_pe, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.097516 ms |
| - | Scoring Baseline | 0.500000 | 2.058476 ms |
| - | Reference Implementation | 0.319183 | 4.343598 ms |
