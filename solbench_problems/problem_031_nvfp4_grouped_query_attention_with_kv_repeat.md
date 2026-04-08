## Description

FP4-quantized Grouped Query Attention (GQA) with KV cache repeat operation. Implements attention with 40 query heads and 8 key-value heads (5x repeat factor). Uses FP4 block-scaled quantization for Q@K^T and Attn@V matrix multiplications with softmax in between. Note: seq_len must be a multiple of 128 for FP4 quantization compatibility.
| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| attn_weights | [batch_size, num_attention_heads, seq_len, seq_len] | bfloat16 |

| # | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 8 | 128 | 5 | 4 | 512 | 4.3368 | 0.0971 |
| 2 | 40 | 8 | 128 | 5 | 64 | 256 | 30.5075 | 0.0119 |
| 3 | 40 | 8 | 128 | 5 | 1 | 2048 | 8.4392 | 0.2655 |
| 4 | 40 | 8 | 128 | 5 | 16 | 512 | 14.1284 | 0.3884 |
| 5 | 40 | 8 | 128 | 5 | 8 | 2048 | 60.0819 | 0.0949 |
| 6 | 40 | 8 | 128 | 5 | 16 | 256 | 8.4085 | 0.1381 |
| 7 | 40 | 8 | 128 | 5 | 32 | 512 | 26.7588 | 0.0237 |
| 8 | 40 | 8 | 128 | 5 | 4 | 256 | 3.0999 | 0.0345 |
| 9 | 40 | 8 | 128 | 5 | 2 | 128 | 2.2221 | 0.0069 |
| 10 | 40 | 8 | 128 | 5 | 8 | 1024 | 17.9956 | 0.6129 |
| 11 | 40 | 8 | 128 | 5 | 16 | 1024 | 34.9610 | 0.0474 |
| 12 | 40 | 8 | 128 | 5 | 32 | 1024 | 68.5240 | 0.0949 |
| 13 | 40 | 8 | 128 | 5 | 1 | 1024 | 3.3551 | 0.0766 |
| 14 | 40 | 8 | 128 | 5 | 8 | 512 | 7.6138 | 0.1942 |
| 15 | 40 | 8 | 128 | 5 | 2 | 1024 | 5.4432 | 0.1532 |
| 16 | 40 | 8 | 128 | 5 | 8 | 128 | 4.3343 | 0.0275 |

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

@torch.no_grad()
def run(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """FP4-quantized GQA with KV repeat."""
    batch_size = query_states.shape[0]
    seq_len = query_states.shape[2]
    num_attention_heads = 40
    num_key_value_groups = 5
    head_dim = 128
    scaling = head_dim ** -0.5
    
    scaler = BlockWiseScalerNVFP4()
    
    # Step 1: Repeat KV heads to match query heads (GQA pattern)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Step 2: Reshape for batch processing
    # For Q @ K^T: Q is [B*heads, seq_len, head_dim], K^T needs to be [B*heads, head_dim, seq_len]
    # CuBLASRefBlockwiseGemm expects mat_a: [B, M, K], mat_b: [B, N, K] and computes A @ B^T
    # So for Q @ K^T: mat_a = Q [B*heads, seq_len, head_dim], mat_b = K [B*heads, seq_len, head_dim]
    # Result: [B*heads, seq_len, seq_len]
    
    q_reshaped = query_states.reshape(
        batch_size * num_attention_heads, seq_len, head_dim
    )
    k_reshaped = key_states.reshape(
        batch_size * num_attention_heads, seq_len, head_dim
    )
    v_reshaped = value_states.reshape(
        batch_size * num_attention_heads, seq_len, head_dim
    )
    
    # Step 3: Quantize Q and K to FP4 with block-wise scaling
    q_fp4, scale_q, global_decode_q = scaler.quantize(q_reshaped)
    k_fp4, scale_k, global_decode_k = scaler.quantize(k_reshaped)
    
    # Step 4: Compute Q @ K^T with FP4 scaled_mm
    # CuBLASRefBlockwiseGemm computes A @ B^T, so we pass Q and K directly
    qk_gemm = CuBLASRefBlockwiseGemm(
        b=batch_size * num_attention_heads,
        m=seq_len,
        n=seq_len,
        k=head_dim
    )
    
    attn_scores = qk_gemm.scaled_mm(
        mat_a=q_fp4,
        mat_b=k_fp4,
        scale_a=scale_q,
        global_decode_a=global_decode_q,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=scale_k,
        global_decode_b=global_decode_k,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    
    # Apply scaling factor
    attn_scores = attn_scores * scaling
    
    # Reshape back to [B, num_heads, seq_len, seq_len]
    attn_scores = attn_scores.reshape(
        batch_size, num_attention_heads, seq_len, seq_len
    )
    
    # Step 5: Apply causal mask
    attn_scores = attn_scores + attention_mask
    
    # Step 6: Softmax (in float32 for numerical stability)
    attn_weights = torch.nn.functional.softmax(
        attn_scores, dim=-1, dtype=torch.float32
    ).to(torch.bfloat16)
    
    # Step 7: Compute Attn @ V with FP4 scaled_mm
    # Attn: [B*heads, seq_len, seq_len], V: [B*heads, seq_len, head_dim]
    # We want Attn @ V = [B*heads, seq_len, head_dim]
    # CuBLASRefBlockwiseGemm computes A @ B^T, so we need to transpose V
    # If we pass Attn and V^T, we get Attn @ (V^T)^T = Attn @ V
    # V^T: [B*heads, head_dim, seq_len]
    
    attn_weights_reshaped = attn_weights.reshape(
        batch_size * num_attention_heads, seq_len, seq_len
    )
    attn_fp4, scale_attn, global_decode_attn = scaler.quantize(attn_weights_reshaped)
    
    # Transpose V: [B*heads, seq_len, head_dim] -> [B*heads, head_dim, seq_len]
    v_transposed = v_reshaped.transpose(1, 2).contiguous()
    v_fp4_t, scale_v_t, global_decode_v_t = scaler.quantize(v_transposed)
    
    av_gemm = CuBLASRefBlockwiseGemm(
        b=batch_size * num_attention_heads,
        m=seq_len,
        n=head_dim,
        k=seq_len
    )
    
    attn_output = av_gemm.scaled_mm(
        mat_a=attn_fp4,
        mat_b=v_fp4_t,
        scale_a=scale_attn,
        global_decode_a=global_decode_attn,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=scale_v_t,
        global_decode_b=global_decode_v_t,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    
    # Reshape back to [B, num_heads, seq_len, head_dim]
    attn_output = attn_output.reshape(
        batch_size, num_attention_heads, seq_len, head_dim
    )
    
    return attn_output, attn_weights
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.076952 ms |
| - | Scoring Baseline | 0.500000 | 10.873128 ms |
| - | Reference Implementation | 0.146116 | 66.265795 ms |
