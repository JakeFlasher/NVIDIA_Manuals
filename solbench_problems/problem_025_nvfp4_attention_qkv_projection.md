## Description

FP4-quantized Q, K, V projections for multi-head attention with GQA (32 query heads, 2 key-value heads). Performs three large matrix multiplications with FP4 weights and block-scaled quantization. Transforms hidden_size (2688) to num_heads * head_dim (4096) for queries and num_key_value_heads * head_dim (256) for keys and values.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| q_weight_fp4 | [q_size, hidden_packed] | float4_e2m1fn_x2 |
| q_scale | [q_size, hidden_sf_k] | float8_e4m3fn |
| q_global_decode_scale | scalar | float32 |
| k_weight_fp4 | [kv_size, hidden_packed] | float4_e2m1fn_x2 |
| k_scale | [kv_size, hidden_sf_k] | float8_e4m3fn |
| k_global_decode_scale | scalar | float32 |
| v_weight_fp4 | [kv_size, hidden_packed] | float4_e2m1fn_x2 |
| v_scale | [kv_size, hidden_sf_k] | float8_e4m3fn |
| v_global_decode_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, seq_len, q_size] | float16 |
| key_states | [batch_size, seq_len, kv_size] | float16 |
| value_states | [batch_size, seq_len, kv_size] | float16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_size | kv_size | sf_vec_size | hidden_packed | hidden_sf_k | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 8 | 512 | 0.6629 | 0.0172 |
| 2 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 1 | 256 | 0.4318 | 0.0011 |
| 3 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 16 | 256 | 0.5786 | 0.0172 |
| 4 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 32 | 256 | 0.9518 | 0.0344 |
| 5 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 32 | 512 | 1.6577 | 0.0687 |
| 6 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 1 | 2048 | 0.5451 | 0.0086 |
| 7 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 1 | 8192 | 0.9580 | 0.0344 |
| 8 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 2 | 256 | 0.5029 | 0.0021 |
| 9 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 32 | 128 | 0.6228 | 0.0172 |
| 10 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 64 | 128 | 1.0146 | 0.0344 |
| 11 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 4 | 512 | 0.5413 | 0.0086 |
| 12 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 4 | 256 | 0.5023 | 0.0043 |
| 13 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 4 | 2048 | 0.9733 | 0.0344 |
| 14 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 1 | 128 | 0.5148 | 0.0005 |
| 15 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 1 | 1024 | 0.5206 | 0.0043 |
| 16 | 2688 | 32 | 2 | 128 | 4096 | 256 | 16 | 1344 | 168 | 2 | 512 | 0.8637 | 0.0043 |

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

def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Generate input tensors for FP4-quantized QKV projections."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 2688
    q_size = 4096
    kv_size = 256
    sf_vec_size = 16
    hidden_packed = hidden_size // 2
    hidden_sf_k = hidden_size // sf_vec_size
    
    scaler = BlockWiseScalerNVFP4()
    
    # Generate input hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Generate and quantize Q weights
    q_weight_fp32 = torch.randn(1, q_size, hidden_size, dtype=torch.float32, device=device)
    q_weight_fp4, q_scale, q_global_decode_scale = scaler.quantize(q_weight_fp32)
    q_weight_fp4 = q_weight_fp4.squeeze(0)
    q_scale = q_scale.squeeze(0)
    
    # Generate and quantize K weights
    k_weight_fp32 = torch.randn(1, kv_size, hidden_size, dtype=torch.float32, device=device)
    k_weight_fp4, k_scale, k_global_decode_scale = scaler.quantize(k_weight_fp32)
    k_weight_fp4 = k_weight_fp4.squeeze(0)
    k_scale = k_scale.squeeze(0)
    
    # Generate and quantize V weights
    v_weight_fp32 = torch.randn(1, kv_size, hidden_size, dtype=torch.float32, device=device)
    v_weight_fp4, v_scale, v_global_decode_scale = scaler.quantize(v_weight_fp32)
    v_weight_fp4 = v_weight_fp4.squeeze(0)
    v_scale = v_scale.squeeze(0)
    
    return {
        "hidden_states": hidden_states,
        "q_weight_fp4": q_weight_fp4,
        "q_scale": q_scale,
        "q_global_decode_scale": q_global_decode_scale.item(),
        "k_weight_fp4": k_weight_fp4,
        "k_scale": k_scale,
        "k_global_decode_scale": k_global_decode_scale.item(),
        "v_weight_fp4": v_weight_fp4,
        "v_scale": v_scale,
        "v_global_decode_scale": v_global_decode_scale.item(),
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    q_global_decode_scale: float,
    k_weight_fp4: torch.Tensor,
    k_scale: torch.Tensor,
    k_global_decode_scale: float,
    v_weight_fp4: torch.Tensor,
    v_scale: torch.Tensor,
    v_global_decode_scale: float,
):
    """FP4-quantized Q, K, V projections for multi-head attention with GQA.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        q_weight_fp4: FP4-quantized Q weights [q_size, hidden_packed]
        q_scale: Block scales for Q weights [q_size, hidden_sf_k]
        q_global_decode_scale: Global decode scale for Q weights (scalar)
        k_weight_fp4: FP4-quantized K weights [kv_size, hidden_packed]
        k_scale: Block scales for K weights [kv_size, hidden_sf_k]
        k_global_decode_scale: Global decode scale for K weights (scalar)
        v_weight_fp4: FP4-quantized V weights [kv_size, hidden_packed]
        v_scale: Block scales for V weights [kv_size, hidden_sf_k]
        v_global_decode_scale: Global decode scale for V weights (scalar)
    
    Returns:
        Tuple of (query_states, key_states, value_states)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    q_size = q_weight_fp4.shape[0]
    kv_size = k_weight_fp4.shape[0]
    
    scaler = BlockWiseScalerNVFP4()
    
    # Convert scalar floats to tensors
    q_global_scale_tensor = torch.tensor(q_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    k_global_scale_tensor = torch.tensor(k_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    v_global_scale_tensor = torch.tensor(v_global_decode_scale, dtype=torch.float32, device=hidden_states.device)
    
    # Quantize input activations to FP4 with block scaling
    hidden_states_fp4, hidden_scale, hidden_global_scale = scaler.quantize(hidden_states)
    # hidden_states_fp4: [batch_size, seq_len, hidden_size // 2]
    # hidden_scale: [batch_size, seq_len, hidden_size // 16]
    
    # Q projection: [batch_size, seq_len, hidden_size] @ [hidden_size, q_size]^T
    q_gemm = CuBLASRefBlockwiseGemm(
        b=batch_size,
        m=seq_len,
        n=q_size,
        k=hidden_size
    )
    
    # Expand weights for batched matmul
    q_weight_transposed = q_weight_fp4.unsqueeze(0).expand(batch_size, -1, -1)
    q_scale_transposed = q_scale.unsqueeze(0).expand(batch_size, -1, -1)
    
    query_states = q_gemm.scaled_mm(
        mat_a=hidden_states_fp4,
        mat_b=q_weight_transposed,
        scale_a=hidden_scale,
        global_decode_a=hidden_global_scale,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=q_scale_transposed,
        global_decode_b=q_global_scale_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32
    )
    
    # K projection: [batch_size, seq_len, hidden_size] @ [hidden_size, kv_size]^T
    k_gemm = CuBLASRefBlockwiseGemm(
        b=batch_size,
        m=seq_len,
        n=kv_size,
        k=hidden_size
    )
    
    k_weight_transposed = k_weight_fp4.unsqueeze(0).expand(batch_size, -1, -1)
    k_scale_transposed = k_scale.unsqueeze(0).expand(batch_size, -1, -1)
    
    key_states = k_gemm.scaled_mm(
        mat_a=hidden_states_fp4,
        mat_b=k_weight_transposed,
        scale_a=hidden_scale,
        global_decode_a=hidden_global_scale,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=k_scale_transposed,
        global_decode_b=k_global_scale_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32
    )
    
    # V projection: [batch_size, seq_len, hidden_size] @ [hidden_size, kv_size]^T
    v_gemm = CuBLASRefBlockwiseGemm(
        b=batch_size,
        m=seq_len,
        n=kv_size,
        k=hidden_size
    )
    
    v_weight_transposed = v_weight_fp4.unsqueeze(0).expand(batch_size, -1, -1)
    v_scale_transposed = v_scale.unsqueeze(0).expand(batch_size, -1, -1)
    
    value_states = v_gemm.scaled_mm(
        mat_a=hidden_states_fp4,
        mat_b=v_weight_transposed,
        scale_a=hidden_scale,
        global_decode_a=hidden_global_scale,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=v_scale_transposed,
        global_decode_b=v_global_scale_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32
    )

    query_states = query_states.to(torch.float16)
    key_states = key_states.to(torch.float16)
    value_states = value_states.to(torch.float16)
    return query_states, key_states, value_states

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"batch_size": 2, "seq_len": 128},
        device=torch.device("cuda:0")
    )
    q, k, v = run(**inputs)
    print(f"Query: {q.shape}, Key: {k.shape}, Value: {v.shape}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.009368 ms |
| - | Scoring Baseline | 0.500000 | 0.690271 ms |
| - | Reference Implementation | 0.215390 | 2.746009 ms |
