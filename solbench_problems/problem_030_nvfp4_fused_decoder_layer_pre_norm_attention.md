## Description

FP4-quantized fused decoder layer pre-normalization and complete attention computation. Combines RMSNorm on input hidden states, Q/K/V projections with FP4 quantization, per-head RMSNorm on Q and K, RoPE positional embeddings, Grouped Query Attention with KV repeat, and output projection with FP4 quantization.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| input_layernorm_weight | [hidden_size] | bfloat16 |
| q_weight_fp4 | [q_out_features, hidden_size_half] | float4_e2m1fn_x2 |
| q_scale | [q_out_features, hidden_scale_blocks] | float8_e4m3fn |
| q_global_scale | scalar | float32 |
| k_weight_fp4 | [kv_out_features, hidden_size_half] | float4_e2m1fn_x2 |
| k_scale | [kv_out_features, hidden_scale_blocks] | float8_e4m3fn |
| k_global_scale | scalar | float32 |
| v_weight_fp4 | [kv_out_features, hidden_size_half] | float4_e2m1fn_x2 |
| v_scale | [kv_out_features, hidden_scale_blocks] | float8_e4m3fn |
| v_global_scale | scalar | float32 |
| o_weight_fp4 | [hidden_size, q_out_half] | float4_e2m1fn_x2 |
| o_scale | [hidden_size, q_out_scale_blocks] | float8_e4m3fn |
| o_global_scale | scalar | float32 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_out_features | kv_out_features | hidden_size_half | q_out_half | kv_out_half | hidden_scale_blocks | q_out_scale_blocks | kv_out_scale_blocks | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 4 | 256 | 2.0153 | 0.0306 |
| 2 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 64 | 128 | 10.3357 | 0.2391 |
| 3 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 2 | 2048 | 8.9070 | 0.1611 |
| 4 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 16 | 512 | 11.5354 | 0.2558 |
| 5 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 8 | 256 | 3.1481 | 0.0612 |
| 6 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 2 | 512 | 2.1266 | 0.0320 |
| 7 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 16 | 1024 | 25.4144 | 0.5558 |
| 8 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 1 | 8192 | 43.4873 | 0.3682 |
| 9 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 1 | 1024 | 2.2968 | 0.0347 |
| 10 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 32 | 256 | 10.6996 | 0.2447 |
| 11 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 32 | 512 | 22.0948 | 0.5115 |
| 12 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 64 | 256 | 20.4769 | 0.4894 |
| 13 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 8 | 512 | 6.1127 | 0.1279 |
| 14 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 1 | 2048 | 4.5975 | 0.0806 |
| 15 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 2 | 256 | 1.5131 | 0.0153 |
| 16 | 5120 | 40 | 8 | 128 | 5120 | 1024 | 2560 | 2560 | 512 | 320 | 320 | 64 | 2 | 4096 | 25.7281 | 0.2371 |

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
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 5120
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    rms_norm_eps = 1e-6
    
    q_out_features = num_attention_heads * head_dim
    kv_out_features = num_key_value_heads * head_dim
    
    scaler = BlockWiseScalerNVFP4()
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device)
    
    input_layernorm_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
    # Scale weights by 1/sqrt(fan_in) to keep output magnitudes reasonable
    q_weight_fp32 = torch.randn(1, q_out_features, hidden_size, dtype=torch.float32, device=device) * (hidden_size ** -0.5)
    q_weight_fp4, q_scale, q_global_scale = scaler.quantize(q_weight_fp32)
    q_weight_fp4 = q_weight_fp4.squeeze(0)
    q_scale = q_scale.squeeze(0)

    k_weight_fp32 = torch.randn(1, kv_out_features, hidden_size, dtype=torch.float32, device=device) * (hidden_size ** -0.5)
    k_weight_fp4, k_scale, k_global_scale = scaler.quantize(k_weight_fp32)
    k_weight_fp4 = k_weight_fp4.squeeze(0)
    k_scale = k_scale.squeeze(0)

    v_weight_fp32 = torch.randn(1, kv_out_features, hidden_size, dtype=torch.float32, device=device) * (hidden_size ** -0.5)
    v_weight_fp4, v_scale, v_global_scale = scaler.quantize(v_weight_fp32)
    v_weight_fp4 = v_weight_fp4.squeeze(0)
    v_scale = v_scale.squeeze(0)

    o_weight_fp32 = torch.randn(1, hidden_size, q_out_features, dtype=torch.float32, device=device) * (q_out_features ** -0.5)
    o_weight_fp4, o_scale, o_global_scale = scaler.quantize(o_weight_fp32)
    o_weight_fp4 = o_weight_fp4.squeeze(0)
    o_scale = o_scale.squeeze(0)
    
    return {
        "hidden_states": hidden_states,
        "cos": cos,
        "sin": sin,
        "attention_mask": attention_mask,
        "input_layernorm_weight": input_layernorm_weight,
        "q_weight_fp4": q_weight_fp4,
        "q_scale": q_scale,
        "q_global_scale": q_global_scale.item(),
        "k_weight_fp4": k_weight_fp4,
        "k_scale": k_scale,
        "k_global_scale": k_global_scale.item(),
        "v_weight_fp4": v_weight_fp4,
        "v_scale": v_scale,
        "v_global_scale": v_global_scale.item(),
        "o_weight_fp4": o_weight_fp4,
        "o_scale": o_scale,
        "o_global_scale": o_global_scale.item(),
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rms_norm_eps": rms_norm_eps,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_weight_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    q_global_scale: float,
    k_weight_fp4: torch.Tensor,
    k_scale: torch.Tensor,
    k_global_scale: float,
    v_weight_fp4: torch.Tensor,
    v_scale: torch.Tensor,
    v_global_scale: float,
    o_weight_fp4: torch.Tensor,
    o_scale: torch.Tensor,
    o_global_scale: float,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
):
    hidden_size = 5120
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    
    batch, seq_len, _ = hidden_states.shape
    scaler = BlockWiseScalerNVFP4()
    
    q_out_features = num_attention_heads * head_dim
    kv_out_features = num_key_value_heads * head_dim
    
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(input_dtype)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(q, k, cos_val, sin_val):
        cos_val = cos_val.unsqueeze(1)
        sin_val = sin_val.unsqueeze(1)
        q_embed = (q * cos_val) + (rotate_half(q) * sin_val)
        k_embed = (k * cos_val) + (rotate_half(k) * sin_val)
        return q_embed, k_embed
    
    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        bs, num_kv_heads, slen, hd = x.shape
        x = x[:, :, None, :, :].expand(bs, num_kv_heads, n_rep, slen, hd)
        return x.reshape(bs, num_kv_heads * n_rep, slen, hd)
    
    def fp4_linear(x, weight_fp4, scale, global_scale, out_features, in_features):
        bs, sl, inf = x.shape
        x_flat = x.reshape(-1, inf)
        
        m_padded = ((bs * sl + 127) // 128) * 128
        pad_rows = m_padded - (bs * sl)
        if pad_rows > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, 0, 0, pad_rows))
        
        x_fp4, x_scale, x_global_scale = scaler.quantize(x_flat.to(torch.float32).unsqueeze(0))
        x_fp4 = x_fp4.squeeze(0)
        x_scale = x_scale.squeeze(0)
        
        gemm = CuBLASRefBlockwiseGemm(b=1, m=m_padded, n=out_features, k=in_features)
        
        output = gemm.scaled_mm(
            mat_a=x_fp4.unsqueeze(0),
            mat_b=weight_fp4.unsqueeze(0),
            scale_a=x_scale.unsqueeze(0),
            global_decode_a=x_global_scale,
            scale_recipe_a=ScalingType.BlockWise1x16,
            scale_b=scale.unsqueeze(0),
            global_decode_b=torch.tensor(global_scale, device=x.device, dtype=torch.float32),
            scale_recipe_b=ScalingType.BlockWise1x16,
            bias=None,
            output_dtype=torch.bfloat16
        )
        
        output = output.squeeze(0)[:bs * sl, :].reshape(bs, sl, out_features)
        return output
    
    hidden_states = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)
    
    query_states = fp4_linear(hidden_states, q_weight_fp4, q_scale, q_global_scale, q_out_features, hidden_size)
    key_states = fp4_linear(hidden_states, k_weight_fp4, k_scale, k_global_scale, kv_out_features, hidden_size)
    value_states = fp4_linear(hidden_states, v_weight_fp4, v_scale, v_global_scale, kv_out_features, hidden_size)
    
    query_states = query_states.view(batch, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch, seq_len, q_out_features)
    
    attn_output = fp4_linear(attn_output, o_weight_fp4, o_scale, o_global_scale, hidden_size, q_out_features)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.133004 ms |
| - | Scoring Baseline | 0.500000 | 7.812551 ms |
| - | Reference Implementation | 0.370082 | 13.299213 ms |
