## Description

FP4-quantized multimodal embedding projection that merges text embeddings with vision features. Takes text token embeddings and vision token embeddings, applies FP4-quantized linear projections to align dimensions, and merges them based on vision token mask positions. Uses block-wise FP4 quantization (16 elements per block) for the projection weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| text_input_ids | [batch_size, seq_len] | int64 |
| text_embedding_weight | [vocab_size, hidden_size] | bfloat16 |
| vision_features | [batch_size, num_vision_tokens, vision_hidden_size] | float32 |
| vision_proj_weight_fp4 | [1, hidden_size, vision_hidden_size_half] | float4_e2m1fn_x2 |
| vision_proj_weight_scale | [1, hidden_size, num_vision_scale_blocks] | float8_e4m3fn |
| vision_proj_weight_global_scale | scalar | float32 |
| vision_proj_bias | [hidden_size] | bfloat16 |
| vision_token_mask | [batch_size, seq_len] | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| merged_embeddings | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | vocab_size | vision_hidden_size | sf_vec_size | vision_hidden_size_half | num_vision_scale_blocks | batch_size | seq_len | num_vision_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 1 | 1024 | 256 | 0.3460 | 0.0831 |
| 2 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 4 | 4096 | 1024 | 0.9307 | 0.1117 |
| 3 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 2 | 8192 | 1280 | 0.7215 | 0.1036 |
| 4 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 8 | 1024 | 384 | 0.7341 | 0.1019 |
| 5 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 2 | 512 | 256 | 0.3569 | 0.0844 |
| 6 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 16 | 2048 | 512 | 1.6434 | 0.1423 |
| 7 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 8 | 2048 | 512 | 0.9351 | 0.1117 |
| 8 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 1 | 256 | 128 | 0.3398 | 0.0820 |
| 9 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 1 | 2048 | 512 | 0.3607 | 0.0850 |
| 10 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 4 | 512 | 256 | 0.3897 | 0.0877 |
| 11 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 32 | 1024 | 384 | 2.2536 | 0.1642 |
| 12 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 32 | 512 | 256 | 1.5442 | 0.1336 |
| 13 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 4 | 1024 | 384 | 0.4749 | 0.0915 |
| 14 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 1 | 4096 | 768 | 0.3769 | 0.0874 |
| 15 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 2 | 2048 | 512 | 0.4065 | 0.0888 |
| 16 | 8192 | 151936 | 3584 | 16 | 1792 | 224 | 2 | 4096 | 1024 | 0.5929 | 0.0964 |

```python
import torch
import math
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
    num_vision_tokens = axes_and_scalars["num_vision_tokens"]
    hidden_size = 8192
    vocab_size = 151936
    vision_hidden_size = 3584
    sf_vec_size = 16
    
    # Text input IDs
    text_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device)
    
    # Text embedding weight
    text_embedding_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Vision features
    vision_features = torch.randn(batch_size, num_vision_tokens, vision_hidden_size, dtype=torch.float32, device=device)
    
    # Create FP4 quantized projection weight
    scaler = BlockWiseScalerNVFP4()
    weight_fp32 = torch.randn(1, hidden_size, vision_hidden_size, dtype=torch.float32, device=device)
    vision_proj_weight_fp4, vision_proj_weight_scale, vision_proj_weight_global_scale = scaler.quantize(weight_fp32)
    
    # Vision projection bias
    vision_proj_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    
    # Vision token mask - randomly place vision tokens
    vision_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for b in range(batch_size):
        num_to_place = min(num_vision_tokens, seq_len)
        positions = torch.randperm(seq_len, device=device)[:num_to_place]
        vision_token_mask[b, positions] = True
    
    return {
        "text_input_ids": text_input_ids,
        "text_embedding_weight": text_embedding_weight,
        "vision_features": vision_features,
        "vision_proj_weight_fp4": vision_proj_weight_fp4,
        "vision_proj_weight_scale": vision_proj_weight_scale,
        "vision_proj_weight_global_scale": vision_proj_weight_global_scale.item(),
        "vision_proj_bias": vision_proj_bias,
        "vision_token_mask": vision_token_mask,
    }

@torch.no_grad()
def run(
    text_input_ids: torch.Tensor,
    text_embedding_weight: torch.Tensor,
    vision_features: torch.Tensor,
    vision_proj_weight_fp4: torch.Tensor,
    vision_proj_weight_scale: torch.Tensor,
    vision_proj_weight_global_scale: float,
    vision_proj_bias: torch.Tensor,
    vision_token_mask: torch.Tensor,
):
    batch_size, seq_len = text_input_ids.shape
    num_vision_tokens = vision_features.shape[1]
    vision_hidden_size = vision_features.shape[2]
    hidden_size = text_embedding_weight.shape[1]
    
    # Text embedding lookup
    text_embeds = torch.nn.functional.embedding(text_input_ids, text_embedding_weight)
    
    # FP4 quantization utilities
    scaler = BlockWiseScalerNVFP4()
    
    # Quantize vision features to FP4
    act_fp4, act_scale, act_global_scale = scaler.quantize(vision_features)
    
    # Prepare weight for batched GEMM (replicate across batch)
    weight_fp4_batched = vision_proj_weight_fp4.expand(batch_size, -1, -1)
    weight_scale_batched = vision_proj_weight_scale.expand(batch_size, -1, -1)
    
    # Initialize GEMM reference
    gemm_ref = CuBLASRefBlockwiseGemm(
        b=batch_size,
        m=num_vision_tokens,
        n=hidden_size,
        k=vision_hidden_size
    )
    
    # Convert global scale to tensor if needed
    global_decode_weight = torch.tensor(vision_proj_weight_global_scale, device=vision_features.device, dtype=torch.float32)
    
    # Perform FP4 matrix multiplication
    vision_embeds = gemm_ref.scaled_mm(
        mat_a=act_fp4,
        mat_b=weight_fp4_batched,
        scale_a=act_scale,
        global_decode_a=act_global_scale,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=weight_scale_batched,
        global_decode_b=global_decode_weight,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.bfloat16
    )
    
    # Add bias
    vision_embeds = vision_embeds + vision_proj_bias.unsqueeze(0).unsqueeze(0)
    
    # Merge text and vision embeddings based on mask
    merged_embeds = text_embeds.clone()
    
    for b in range(batch_size):
        vision_positions = torch.where(vision_token_mask[b])[0]
        if len(vision_positions) > 0:
            num_vision = min(len(vision_positions), vision_embeds.shape[1])
            merged_embeds[b, vision_positions[:num_vision]] = vision_embeds[b, :num_vision]
    
    return merged_embeds
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.101193 ms |
| - | Scoring Baseline | 0.500000 | 0.633377 ms |
| - | Reference Implementation | 0.223515 | 1.950217 ms |
