## Description

FP8-quantized SSM parameter discretization in Mamba2 that computes dt (time step), dA (discretized A matrix), and dB (discretized B matrix) using FP8 linear projections with blockwise scaling.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, num_heads] | bfloat16 |
| B | [batch_size, seq_len, n_groups, ssm_state_size] | bfloat16 |
| dt_proj_weight | [num_heads, num_heads] | bfloat16 |
| dt_bias | [num_heads] | bfloat16 |
| A_log | [num_heads] | float32 |
| time_step_limit_min | scalar | float32 |
| time_step_limit_max | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| dt | [batch_size, seq_len, num_heads, head_dim] | bfloat16 |
| dA | [batch_size, seq_len, num_heads, head_dim, ssm_state_size] | float32 |
| dB | [batch_size, seq_len, num_heads, head_dim, ssm_state_size] | float32 |

| # | num_heads | head_dim | ssm_state_size | n_groups | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 128 | 128 | 8 | 16 | 128 | 0.0899 | 0.0000 |
| 2 | 128 | 128 | 128 | 8 | 4 | 256 | 0.0697 | 0.0000 |
| 3 | 128 | 128 | 128 | 8 | 2 | 512 | 0.0685 | 0.0000 |
| 4 | 128 | 128 | 128 | 8 | 2 | 128 | 0.0560 | 0.0000 |
| 5 | 128 | 128 | 128 | 8 | 4 | 128 | 0.0601 | 0.0000 |
| 6 | 128 | 128 | 128 | 8 | 8 | 128 | 0.0718 | 0.0000 |

```python
import torch
import torch.nn.functional as F
import torch

from enum import StrEnum

class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.
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
    """

    E4M3_MAX = 448.0

    def __init__(self, scaling_type: ScalingType):
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape

        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }

        self.block_size_m, self.block_size_k = scaling_map[scaling_type]
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute scale factors based on the scaling type."""
        if self.scaling_type == ScalingType.TensorWise:
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        assert M % self.block_size_m == 0
        assert K % self.block_size_k == 0

        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        """Apply scaling to tensor based on the scaling type."""
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
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
    """Reference implementation of blockwise-scaled GEMM via dequantize-then-matmul."""

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
        """Scaled matrix multiplication: dequantize A and B, then matmul in float32."""
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_heads = axes_and_scalars["num_heads"]
    head_dim = axes_and_scalars["head_dim"]
    ssm_state_size = axes_and_scalars["ssm_state_size"]
    n_groups = axes_and_scalars["n_groups"]
    
    hidden_states = torch.randn(batch_size, seq_len, num_heads, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch_size, seq_len, n_groups, ssm_state_size, dtype=torch.bfloat16, device=device)
    dt_proj_weight = torch.randn(num_heads, num_heads, dtype=torch.bfloat16, device=device)
    dt_bias = torch.ones(num_heads, dtype=torch.bfloat16, device=device)
    A_log = torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32, device=device))
    
    return {
        "hidden_states": hidden_states,
        "B": B,
        "dt_proj_weight": dt_proj_weight,
        "dt_bias": dt_bias,
        "A_log": A_log,
        "time_step_limit_min": 0.0,
        "time_step_limit_max": float('inf'),
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    time_step_limit_min: float,
    time_step_limit_max: float,
):
    """
    FP8-quantized SSM parameter discretization for Mamba2.

    REWRITTEN for compact computation:
    - dA computed at [B,L,H] instead of [B,L,H,hd,ssm]
    - dB computed at [B,L,H,ssm] via group-wise multiply (no repeat_interleave)
    - Final expand() at return boundary (zero-copy views)
    """
    batch_size, seq_len, num_heads = hidden_states.shape
    head_dim = 128
    ssm_state_size = 128
    n_groups = 8
    
    # ---- GEMM with FP8 quantization: unchanged from original ----
    hidden_states_flat = hidden_states.reshape(-1, num_heads)

    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    hidden_states_fp32 = hidden_states_flat.to(torch.float32)
    weight_fp32 = dt_proj_weight.to(torch.float32)

    scale_x = activation_scaler.compute_scales(hidden_states_fp32)
    weight_fp32_t = weight_fp32.T
    scales_w = weight_scaler.compute_scales(weight_fp32_t)

    x_scaled = activation_scaler.apply_scaling(
        hidden_states_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    w_scaled = weight_scaler.apply_scaling(
        weight_fp32_t, scales_w, inverse=False, clamp_to_fp8_range=True
    )

    qx = x_scaled.to(torch.float8_e4m3fn)
    qw = w_scaled.T.to(torch.float8_e4m3fn)
    scale_w_cublas = scales_w.T.contiguous()

    dt_proj = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=dt_bias,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    dt_proj = dt_proj.reshape(batch_size, seq_len, num_heads)

    # ---- Softplus + clamp: unchanged ----
    dt = F.softplus(dt_proj.float())
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)

    # ---- COMPACT dA: [B,L,H] instead of [B,L,H,hd,ssm] ----
    # A is [H] — constant across head_dim and ssm_state_size.
    # exp(dt * A) produces identical values along those dims.
    A = -torch.exp(A_log.float())                     # [H]
    dA_compact = torch.exp(dt * A)                    # [B, L, H]

    # ---- COMPACT dB: [B,L,H,ssm] via group-wise multiply ----
    # B is [B,L,G,ssm]. Instead of repeat_interleave (physical copy),
    # reshape dt into groups and broadcast-multiply.
    heads_per_group = num_heads // n_groups
    dt_groups = dt.reshape(batch_size, seq_len, n_groups, heads_per_group)  # [B,L,G,H/G]
    dB_compact = (
        dt_groups.unsqueeze(-1) * B.unsqueeze(-2)     # [B,L,G,H/G,ssm]
    ).reshape(batch_size, seq_len, num_heads, ssm_state_size)  # [B,L,H,ssm]

    # ---- EXPAND via views at return (zero-copy, zero memory traffic) ----
    dt_out = dt.to(torch.bfloat16).unsqueeze(-1).expand(
        -1, -1, -1, head_dim
    )                                                  # [B,L,H,hd] view
    dA = dA_compact.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, head_dim, ssm_state_size
    )                                                  # [B,L,H,hd,ssm] view
    dB = dB_compact.unsqueeze(3).expand(
        -1, -1, -1, head_dim, -1
    )                                                  # [B,L,H,hd,ssm] view

    return dt_out, dA, dB
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000019 ms |
| - | Scoring Baseline | 0.500000 | 0.068540 ms |
| - | Reference Implementation | 0.137264 | 0.437676 ms |
