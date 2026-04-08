## Description

FP4-quantized MoE router projection that computes expert selection scores. Projects hidden states from hidden_size (2688) to n_routed_experts (128) using FP4 weights with block-scaled quantization, followed by sigmoid activation, bias correction, and top-k selection with normalized routing weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | float32 |
| weight_fp4 | [n_routed_experts, hidden_size_half] | float4_e2m1fn_x2 |
| weight_scale | [n_routed_experts, sf_k] | float8_e4m3fn |
| weight_global_scale | scalar | float32 |
| e_score_correction_bias | [n_routed_experts] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| topk_indices | [num_tokens, num_experts_per_tok] | int64 |
| topk_weights | [num_tokens, num_experts_per_tok] | float32 |

| # | hidden_size | n_routed_experts | num_experts_per_tok | n_group | topk_group | sf_vec_size | hidden_size_half | sf_k | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 2560 | 0.2796 | 0.0088 |
| 2 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 2432 | 0.2734 | 0.0084 |
| 3 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 2304 | 0.2650 | 0.0079 |
| 4 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 1280 | 0.2719 | 0.0044 |
| 5 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3200 | 0.3024 | 0.0110 |
| 6 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 384 | 0.2667 | 0.0013 |
| 7 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 1920 | 0.2680 | 0.0066 |
| 8 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3968 | 0.3328 | 0.0136 |
| 9 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3840 | 0.3263 | 0.0132 |
| 10 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 640 | 0.2961 | 0.0022 |
| 11 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 1536 | 0.2644 | 0.0053 |
| 12 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3584 | 0.3198 | 0.0123 |
| 13 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3456 | 0.3104 | 0.0119 |
| 14 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 2048 | 0.2680 | 0.0070 |
| 15 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 1408 | 0.2654 | 0.0048 |
| 16 | 2688 | 128 | 6 | 1 | 1 | 16 | 1344 | 168 | 3072 | 0.3098 | 0.0106 |

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
    """Generate input tensors for NVFP4 MoE router projection."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 2688
    n_routed_experts = 128
    sf_vec_size = 16
    
    # Generate random input hidden states
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.float32,
        device=device
    )
    
    # Initialize FP4 utilities
    scaler = BlockWiseScalerNVFP4()
    
    # Initialize FP32 weight for quantization
    weight_fp32 = torch.randn(
        1,
        n_routed_experts,
        hidden_size,
        dtype=torch.float32,
        device=device
    )
    
    # Quantize weight to FP4 with block scaling
    weight_fp4, weight_scale, weight_global_scale = scaler.quantize(weight_fp32)
    
    # Remove batch dimension
    weight_fp4 = weight_fp4.squeeze(0)  # (n_routed_experts, hidden_size//2)
    weight_scale = weight_scale.squeeze(0)  # (n_routed_experts, sf_k)
    
    # Extract scalar value from tensor
    weight_global_scale_scalar = float(weight_global_scale.item())
    
    # Bias correction for expert load balancing
    e_score_correction_bias = torch.zeros(n_routed_experts, dtype=torch.float32, device=device)
    
    return {
        "hidden_states": hidden_states,
        "weight_fp4": weight_fp4,
        "weight_scale": weight_scale,
        "weight_global_scale": weight_global_scale_scalar,
        "e_score_correction_bias": e_score_correction_bias,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: float,
    e_score_correction_bias: torch.Tensor,
):
    """FP4-quantized MoE router projection with top-k selection.
    
    Args:
        hidden_states: Input tensor of shape (num_tokens, hidden_size)
        weight_fp4: FP4-quantized weight of shape (n_routed_experts, hidden_size//2)
        weight_scale: Block-wise scales of shape (n_routed_experts, sf_k)
        weight_global_scale: Global decode scale (scalar float)
        e_score_correction_bias: Bias correction of shape (n_routed_experts,)
    
    Returns:
        Tuple of (topk_indices, topk_weights)
    """
    num_tokens, hidden_size = hidden_states.shape
    n_routed_experts = weight_fp4.shape[0]
    num_experts_per_tok = 6
    n_group = 1
    topk_group = 1
    routed_scaling_factor = 2.5
    sf_vec_size = 16
    sf_k = hidden_size // sf_vec_size
    
    device = hidden_states.device
    
    # Initialize FP4 utilities
    scaler = BlockWiseScalerNVFP4()
    
    # Quantize input activations to FP4 with block scaling
    # Add batch dimension for quantizer
    act_fp4, act_scale, act_global_scale = scaler.quantize(
        hidden_states.unsqueeze(0).to(torch.float32)
    )
    
    # Add batch dimension to weight for GEMM
    weight_fp4_batched = weight_fp4.unsqueeze(0)  # (1, n_routed_experts, hidden_size//2)
    weight_scale_batched = weight_scale.unsqueeze(0)  # (1, n_routed_experts, sf_k)
    
    # Convert scalar to tensor for GEMM
    weight_global_scale_tensor = torch.tensor(weight_global_scale, dtype=torch.float32, device=device)
    
    # Create GEMM operator
    gemm_ref = CuBLASRefBlockwiseGemm(
        b=1,
        m=num_tokens,
        n=n_routed_experts,
        k=hidden_size
    )
    
    # Perform FP4 matrix multiplication: hidden_states @ weight.T
    router_logits = gemm_ref.scaled_mm(
        mat_a=act_fp4,
        mat_b=weight_fp4_batched,
        scale_a=act_scale,
        global_decode_a=act_global_scale,
        scale_recipe_a=ScalingType.BlockWise1x16,
        scale_b=weight_scale_batched,
        global_decode_b=weight_global_scale_tensor,
        scale_recipe_b=ScalingType.BlockWise1x16,
        bias=None,
        output_dtype=torch.float32,
    )
    
    # Remove batch dimension
    router_logits = router_logits.squeeze(0)  # (num_tokens, n_routed_experts)
    
    # Apply sigmoid activation to get routing scores
    scores = torch.sigmoid(router_logits)
    
    # Apply bias correction for expert load balancing
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    
    # Grouped routing: compute group scores and select top groups
    group_scores = scores_for_choice.view(
        num_tokens,
        n_group,
        n_routed_experts // n_group
    )
    
    # Select top-2 experts per group and sum their scores
    group_scores_sum = group_scores.topk(2, dim=-1)[0].sum(dim=-1)
    
    # Select top groups based on summed scores
    group_idx = torch.topk(
        group_scores_sum,
        k=topk_group,
        dim=-1,
        sorted=False
    )[1]
    
    # Create group mask
    group_mask = torch.zeros_like(group_scores_sum)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand mask to expert level
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_routed_experts // n_group)
        .reshape(num_tokens, n_routed_experts)
    )
    
    # Mask out experts from non-selected groups
    scores_masked = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    
    # Select top-k experts from masked scores
    topk_weights, topk_indices = torch.topk(
        scores_masked,
        k=num_experts_per_tok,
        dim=-1,
        sorted=False
    )
    
    # Normalize top-k weights
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_weights / denominator
    
    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    
    return topk_indices, topk_weights

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"num_tokens": 1024},
        device=torch.device("cuda:0")
    )
    topk_indices, topk_weights = run(**inputs)
    print(f"topk_indices shape: {topk_indices.shape}")
    print(f"topk_weights shape: {topk_weights.shape}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.006906 ms |
| - | Scoring Baseline | 0.500000 | 0.287792 ms |
| - | Reference Implementation | 0.195441 | 1.165923 ms |
