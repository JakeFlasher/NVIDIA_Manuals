## Description

FP4-quantized attention QKV projection with RMSNorm, RoPE application, and KV cache update. Fuses Q/K/V projections (FP4 block-scaled), per-head RMSNorm on Q and K, rotary position embeddings, for GQA attention with 32 query heads and 8 KV heads.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| q_proj_weight_fp4 | [q_size, hidden_size_half] | float4_e2m1fn_x2 |
| q_proj_scale | [q_size, hidden_blocks] | float8_e4m3fn |
| q_proj_global_decode_scale | scalar | float32 |
| k_proj_weight_fp4 | [kv_size, hidden_size_half] | float4_e2m1fn_x2 |
| k_proj_scale | [kv_size, hidden_blocks] | float8_e4m3fn |
| k_proj_global_decode_scale | scalar | float32 |
| v_proj_weight_fp4 | [kv_size, hidden_size_half] | float4_e2m1fn_x2 |
| v_proj_scale | [kv_size, hidden_blocks] | float8_e4m3fn |
| v_proj_global_decode_scale | scalar | float32 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_size | kv_size | hidden_size_half | q_size_half | kv_size_half | hidden_blocks | q_blocks | kv_blocks | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 4 | 512 | 0.8494 | 0.0139 |
| 2 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 4 | 4096 | 2.5655 | 0.1111 |
| 3 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 2 | 2048 | 0.8914 | 0.0278 |
| 4 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 8 | 256 | 0.6860 | 0.0139 |
| 5 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 2 | 512 | 0.6857 | 0.0069 |
| 6 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 1 | 8192 | 1.3931 | 0.0555 |
| 7 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 8 | 1024 | 1.4279 | 0.0555 |
| 8 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 4 | 128 | 0.7003 | 0.0035 |
| 9 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 1 | 512 | 0.7108 | 0.0035 |
| 10 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 32 | 128 | 0.8713 | 0.0278 |
| 11 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 16 | 256 | 0.8731 | 0.0278 |
| 12 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 2 | 4096 | 1.4259 | 0.0555 |
| 13 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 1 | 1024 | 0.6985 | 0.0069 |
| 14 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 1 | 4096 | 0.8744 | 0.0278 |
| 15 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 1 | 2048 | 0.7028 | 0.0139 |
| 16 | 4096 | 32 | 8 | 128 | 4096 | 1024 | 2048 | 2048 | 512 | 256 | 256 | 64 | 2 | 128 | 0.7021 | 0.0017 |

```python
import torch
# --- inlined nvfp4_reference ---
import math
from enum import StrEnum

import torch

class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K).
    """

    BlockWise1x16 = "BlockWise1x16"

class BlockWiseScalerNVFP4:
    def __init__(self, scaling_type: ScalingType = ScalingType.BlockWise1x16):
        self.scaling_type = scaling_type
        self.sf_vec_size = 16
        self.E2M1FN_MAX = 6.0

        # for clamping to nonzero values
        self.E4M3FN_MIN_POS = 2 ** (-9)

    def quantize(self, tensor: torch.Tensor):
        """
        Quantize a high precision tensor to FP4 E2M1 with 1x16 block-wise scaling.

        Args:
            tensor: Input tensor of shape (b, rows, cols) with dtype >= 16 bits (fp16, bf16, fp32)

        Returns:
            Tuple of (quantized_data, scale_factors, global_decode_scale):
            - quantized_data: float4_e2m1fn_x2 tensor of shape (b, rows, cols//2) (2 FP4 values per uint8 byte)
            - scale_factors: FP8 E4M3FN tensor of shape (b, rows, cols//16) with decode scales
            - global_decode_scale: Scalar tensor for global dequantization scaling
        """
        # Compute global amax for the entire tensor
        global_amax = torch.max(torch.abs(tensor)).to(torch.float32)

        # Create scale factors using the helper method (with global scaling)
        decode_scale, global_decode_scale = self._create_scale_factors(
            tensor, global_amax
        )
        # decode_scale shape: (b, rows, cols//16)

        # Now compute encode scale for quantization
        encode_scale = torch.clamp(
            torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
            max=torch.tensor(
                torch.finfo(torch.float32).max,
                device=tensor.device,
                dtype=torch.float32,
            ),
        )
        # encode_scale shape: (b, rows, cols//16)

        # Get tensor dimensions
        b, rows, cols = tensor.shape

        # Reshape to blocks:
        # Scale and quantize
        scaled_x = tensor.view(b, rows, cols // self.sf_vec_size, self.sf_vec_size).to(
            torch.float32
        ) * encode_scale.unsqueeze(-1)
        clipped_x = torch.clamp(scaled_x, -self.E2M1FN_MAX, self.E2M1FN_MAX)

        # Cast to FP4 (packed format): (b, rows, cols//2)
        quantized_data = self._cast_to_fp4x2(clipped_x.reshape(b, rows, cols))

        return quantized_data, decode_scale, global_decode_scale

    def _cast_to_fp4x2(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte).

        Args:
            x: Input tensor of shape (b, rows, cols) with values in range [-6, 6]

        Returns:
            uint8 tensor of shape (b, rows, cols//2) with packed FP4 values
        """
        # FP4 E2M1 encoding
        result = torch.zeros_like(x, dtype=torch.uint8)

        # Positive values
        result[(x >= 0.0) & (x <= 0.25)] = 0
        result[(x > 0.25) & (x < 0.75)] = 1
        result[(x >= 0.75) & (x <= 1.25)] = 2
        result[(x > 1.25) & (x < 1.75)] = 3
        result[(x >= 1.75) & (x <= 2.5)] = 4
        result[(x > 2.5) & (x < 3.5)] = 5
        result[(x >= 3.5) & (x <= 5.0)] = 6
        result[x > 5.0] = 7

        # Negative values
        result[(x >= -0.25) & (x < 0.0)] = 8
        result[(x < -0.25) & (x > -0.75)] = 9
        result[(x <= -0.75) & (x >= -1.25)] = 10
        result[(x < -1.25) & (x > -1.75)] = 11
        result[(x <= -1.75) & (x >= -2.5)] = 12
        result[(x < -2.5) & (x > -3.5)] = 13
        result[(x <= -3.5) & (x >= -5.0)] = 14
        result[x < -5.0] = 15

        # Pack two FP4 values into one byte along cols dimension (dim 1)
        # Input shape: (b, rows, colls)
        # first value (even cols) in low nibble, second value (odd cols) in high nibble
        packed = result[..., ::2] + result[..., 1::2] * 16
        return packed.view(torch.float4_e2m1fn_x2)

    def _create_scale_factors(self, tensor: torch.Tensor, global_amax: torch.Tensor):
        """
        Create scale factors for a given tensor with optional global scaling.

        Args:
            tensor: Input tensor of shape (b, rows, cols)
            global_amax: Optional global max absolute value for global scaling

        Returns:
            Tuple of (decode_scales, global_decode_scale):
            - decode_scales: FP8 E4M3FN tensor of shape (b, rows, cols // 16)
            - global_decode_scale: Scalar tensor (1.0 if no global scaling applied)
        """
        if tensor.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {tensor.dim()}")
        b, rows, cols = tensor.shape

        if cols % self.sf_vec_size != 0:
            raise ValueError(
                f"Cols must be a multiple of {self.sf_vec_size}, got {cols}"
            )

        # Compute max over the block dimensions
        blocked = tensor.view(b, rows, cols // self.sf_vec_size, self.sf_vec_size)
        block_max = blocked.abs().amax(dim=-1)  # (b, rows, cols//16)

        # Compute decode scale: block_max / E2M1_MAX
        decode_scale = block_max / self.E2M1FN_MAX
        decode_scale = decode_scale.to(torch.float32)

        # Apply global scaling (matches nvfp4_tekit.py logic)
        FLOAT8_E4M3_MAX = torch.tensor(448.0, device=tensor.device, dtype=torch.float32)

        # Global encode scale to fit decode scales into FP8 range
        global_encode_scale = torch.div(FLOAT8_E4M3_MAX * self.E2M1FN_MAX, global_amax)
        global_encode_scale = torch.clamp(
            global_encode_scale,
            max=torch.tensor(
                torch.finfo(torch.float32).max,
                device=tensor.device,
                dtype=torch.float32,
            ),
        )

        # Avoid division by zero
        if global_encode_scale == 0.0:
            global_encode_scale = torch.tensor(
                1.0, device=tensor.device, dtype=torch.float32
            )

        global_decode_scale = torch.div(1.0, global_encode_scale)

        # Scale decode_scale to fit in FP8 E4M3FN range
        decode_scale = decode_scale * global_encode_scale
        decode_scale = torch.clamp(
            decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX
        )

        # Convert to FP8 E4M3FN - keep batch-first format (b, rows, cols//16)
        decode_scale = decode_scale.to(dtype=torch.float8_e4m3fn)

        return decode_scale, global_decode_scale

    def convert_to_blocked_format_for_pytorch_scaled_mm(
        self, scale_factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert scale factor tensor to the "blocked" layout expected by torch._scaled_mm.

        Args:
            scale_factors: FP8_E4M3FN Tensor of shape (k//16, m_or_n, b) with strides ((k//16)*m_or_n, m_or_n, 1).

        NOTE: This assumes `rows` is a multiple of 128 and `cols` is a multiple of 4.
        """
        if scale_factors.dim() != 2:
            raise ValueError(
                f"Scale factors must have 2 dimensions, got {scale_factors.dim()}"
            )
        rows, cols = scale_factors.shape
        if rows % 128 != 0:
            raise ValueError(f"Rows must be a multiple of 128, got {rows}")
        if cols % 4 != 0:
            raise ValueError(f"Cols must be a multiple of 4, got {cols}")
        n_row_blocks = math.ceil(rows / 128)
        n_col_blocks = math.ceil(cols / 4)

        blocks = scale_factors.view(n_row_blocks, 128, n_col_blocks, 4).permute(
            0, 2, 1, 3
        )
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        return rearranged.flatten()

class CuBLASRefBlockwiseGemm:
    def __init__(self, b: int, m: int, n: int, k: int):
        self.b = b
        self.m = m
        self.n = n
        self.k = k
        self.sf_vec_size = 16
        self.sf_k = k // self.sf_vec_size
        self.scaler = BlockWiseScalerNVFP4()

        if self.k % self.sf_vec_size != 0:
            raise ValueError(
                f"K must be a multiple of {self.sf_vec_size}, got {self.k}"
            )
        if self.m % 128 != 0:
            raise ValueError(f"M must be a multiple of 128, got {self.m}")
        if self.n % 128 != 0:
            raise ValueError(f"N must be a multiple of 128, got {self.n}")

    def _validate_shapes(self, mat_a: torch.Tensor, mat_b: torch.Tensor):
        if mat_a.dim() != 3:
            raise ValueError(
                f"mat_a must have 3 dimensions (batch, m, k), got {mat_a.dim()}"
            )
        if mat_b.dim() != 3:
            raise ValueError(
                f"mat_b must have 3 dimensions (batch, n, k), got {mat_b.dim()}"
            )

        # the e2m1 tensor has 2 fp4 values per byte, so the shape is (b, m, k//2)
        if mat_a.shape != (self.b, self.m, self.k // 2):
            raise ValueError(
                f"mat_a must have shape (B, M, K//2) = ({self.b}, {self.m}, {self.k // 2}), got {mat_a.shape}"
            )
        if mat_b.shape != (self.b, self.n, self.k // 2):
            raise ValueError(
                f"mat_b must have shape (B, N, K//2) = ({self.b}, {self.n}, {self.k // 2}), got {mat_b.shape}"
            )

    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        global_decode_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        global_decode_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ):
        """
        Perform scaled matrix multiplication: C = (A * scale_a * global_decode_a) @ (B * scale_b * global_decode_b)

        Args:
            mat_a: Quantized matrix A of shape (b, m, k//2)
            mat_b: Quantized matrix B of shape (b, n, k//2)
            scale_a: FP8 E4M3FN scale factors for A of shape (b, m, k//16)
            scale_recipe_a: Scaling type for A (must be BlockWise1x16)
            scale_b: FP8 E4M3FN scale factors for B of shape (b, n, k//16)
            scale_recipe_b: Scaling type for B (must be BlockWise1x16)
            global_decode_a: Optional global decode scale for A (scalar tensor)
            global_decode_b: Optional global decode scale for B (scalar tensor)
            bias: Optional bias tensor (not supported yet)
            output_dtype: Output data type (default: float32)

        Returns:
            Output tensor of shape (b, m, n)
        """
        self._validate_shapes(mat_a, mat_b)

        if (
            scale_recipe_a != ScalingType.BlockWise1x16
            or scale_recipe_b != ScalingType.BlockWise1x16
        ):
            raise ValueError(
                f"Only BlockWise1x16 is supported for scale_recipe, but got {scale_recipe_a} and {scale_recipe_b}"
            )
        if bias is not None:
            raise ValueError("Bias is not supported yet.")

        # Compute combined global scale: global_decode_a * global_decode_b
        global_scale = global_decode_a * global_decode_b

        out = torch.empty(
            (self.m, self.n, self.b), device=mat_a.device, dtype=output_dtype
        )

        mat_a = mat_a.permute(1, 2, 0)
        mat_b = mat_b.permute(1, 2, 0)
        scale_a = scale_a.permute(1, 2, 0)
        scale_b = scale_b.permute(1, 2, 0)

        for b in range(self.b):
            # Extract scales for current batch: (m, k//16, b) -> (m, k//16)
            scale_a_blocked = (
                self.scaler.convert_to_blocked_format_for_pytorch_scaled_mm(
                    scale_a[:, :, b]
                )
            )
            scale_b_blocked = (
                self.scaler.convert_to_blocked_format_for_pytorch_scaled_mm(
                    scale_b[:, :, b]
                )
            )

            # Perform scaled matmul with block-wise scales
            result = torch._scaled_mm(
                mat_a[:, :, b],
                mat_b[:, :, b].transpose(0, 1),  # (n, k) -> (k, n) without moving data
                scale_a_blocked,
                scale_b_blocked,
                bias=None,
                out_dtype=output_dtype,
            )

            # Apply global decode scale: result * global_decode_a * global_decode_b
            out[:, :, b] = result * global_scale

        return out.permute(2, 0, 1)

# --- end inlined nvfp4_reference ---

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for FP4-quantized QKV projection with RoPE."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    q_size = num_attention_heads * head_dim  # 4096
    kv_size = num_key_value_heads * head_dim  # 1024
    sf_vec_size = 16
    
    scaler = BlockWiseScalerNVFP4()
    
    # Input hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # RoPE cos/sin
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    
    # Q projection weights (quantized)
    q_weight_fp32 = torch.randn(1, q_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    q_fp4, q_scale, q_global = scaler.quantize(q_weight_fp32)
    q_proj_weight_fp4 = q_fp4.squeeze(0)
    q_proj_scale = q_scale.squeeze(0)
    q_proj_global_decode_scale = q_global.item()
    
    # K projection weights (quantized) - pad to 1024 (multiple of 128)
    k_weight_fp32 = torch.randn(1, kv_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    k_fp4, k_scale, k_global = scaler.quantize(k_weight_fp32)
    k_proj_weight_fp4 = k_fp4.squeeze(0)
    k_proj_scale = k_scale.squeeze(0)
    k_proj_global_decode_scale = k_global.item()
    
    # V projection weights (quantized) - pad to 1024 (multiple of 128)
    v_weight_fp32 = torch.randn(1, kv_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    v_fp4, v_scale, v_global = scaler.quantize(v_weight_fp32)
    v_proj_weight_fp4 = v_fp4.squeeze(0)
    v_proj_scale = v_scale.squeeze(0)
    v_proj_global_decode_scale = v_global.item()
    
    # RMSNorm weights
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
    return {
        "hidden_states": hidden_states,
        "cos": cos,
        "sin": sin,
        "q_proj_weight_fp4": q_proj_weight_fp4,
        "q_proj_scale": q_proj_scale,
        "q_proj_global_decode_scale": q_proj_global_decode_scale,
        "k_proj_weight_fp4": k_proj_weight_fp4,
        "k_proj_scale": k_proj_scale,
        "k_proj_global_decode_scale": k_proj_global_decode_scale,
        "v_proj_weight_fp4": v_proj_weight_fp4,
        "v_proj_scale": v_proj_scale,
        "v_proj_global_decode_scale": v_proj_global_decode_scale,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rms_norm_eps": 1e-6,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_proj_weight_fp4: torch.Tensor,
    q_proj_scale: torch.Tensor,
    q_proj_global_decode_scale: float,
    k_proj_weight_fp4: torch.Tensor,
    k_proj_scale: torch.Tensor,
    k_proj_global_decode_scale: float,
    v_proj_weight_fp4: torch.Tensor,
    v_proj_scale: torch.Tensor,
    v_proj_global_decode_scale: float,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """FP4-quantized QKV projection with RMSNorm and RoPE."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    sf_vec_size = 16
    
    scaler = BlockWiseScalerNVFP4()
    dtype = torch.bfloat16
    
    # Flatten hidden states for GEMM: [batch*seq_len, hidden_size]
    hidden_flat = hidden_states.view(1, batch_size * seq_len, hidden_size)
    
    # Quantize input activations
    act_fp4, act_scale, act_global = scaler.quantize(hidden_flat.to(torch.float32))
    
    # Convert global decode scales to tensors
    q_global_tensor = torch.tensor(q_proj_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    k_global_tensor = torch.tensor(k_proj_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    v_global_tensor = torch.tensor(v_proj_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    
    # Q projection GEMM
    q_gemm = CuBLASRefBlockwiseGemm(b=1, m=batch_size * seq_len, n=q_size, k=hidden_size)
    query_states = q_gemm.scaled_mm(
        mat_a=act_fp4,
        mat_b=q_proj_weight_fp4.unsqueeze(0),
        scale_a=act_scale,
        global_decode_a=act_global,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=q_proj_scale.unsqueeze(0),
        global_decode_b=q_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        output_dtype=dtype,
    ).squeeze(0)  # [batch*seq_len, q_size]
    
    # K projection GEMM
    k_gemm = CuBLASRefBlockwiseGemm(b=1, m=batch_size * seq_len, n=kv_size, k=hidden_size)
    key_states = k_gemm.scaled_mm(
        mat_a=act_fp4,
        mat_b=k_proj_weight_fp4.unsqueeze(0),
        scale_a=act_scale,
        global_decode_a=act_global,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=k_proj_scale.unsqueeze(0),
        global_decode_b=k_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        output_dtype=dtype,
    ).squeeze(0)  # [batch*seq_len, kv_size]
    
    # V projection GEMM
    v_gemm = CuBLASRefBlockwiseGemm(b=1, m=batch_size * seq_len, n=kv_size, k=hidden_size)
    value_states = v_gemm.scaled_mm(
        mat_a=act_fp4,
        mat_b=v_proj_weight_fp4.unsqueeze(0),
        scale_a=act_scale,
        global_decode_a=act_global,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=v_proj_scale.unsqueeze(0),
        global_decode_b=v_global_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        output_dtype=dtype,
    ).squeeze(0)  # [batch*seq_len, kv_size]
    
    # Reshape to [batch, seq_len, num_heads, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Apply per-head RMSNorm to Q and K
    def rms_norm(x, weight, eps):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed).to(x.dtype)
    
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    # Transpose to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    query_states = (query_states * cos_expanded) + (rotate_half(query_states) * sin_expanded)
    key_states = (key_states * cos_expanded) + (rotate_half(key_states) * sin_expanded)
    
    return query_states, key_states, value_states

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"batch_size": 2, "seq_len": 128},
        device=torch.device("cuda:0")
    )
    q, k, v = run(**inputs)
    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.016514 ms |
| - | Scoring Baseline | 0.500000 | 0.926682 ms |
| - | Reference Implementation | 0.278045 | 2.406005 ms |
