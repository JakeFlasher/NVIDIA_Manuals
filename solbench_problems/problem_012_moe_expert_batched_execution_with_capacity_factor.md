## Description

Optimized batched MoE expert execution with capacity-based token batching. Groups tokens by assigned experts using capacity factors, executes all experts in parallel using batched matrix operations, and performs efficient scatter-gather for weighted aggregation. Uses SwiGLU activation (silu(gate) * up) for expert computation. Capacity factor is fixed at 1.25.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| selected_experts | [num_tokens, num_experts_per_tok] | int64 |
| routing_weights | [num_tokens, num_experts_per_tok] | bfloat16 |
| expert_gate_weights | [num_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| expert_up_weights | [num_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| expert_down_weights | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | num_experts | num_experts_per_tok | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6144 | 2560 | 160 | 8 | 4096 | 5.3862 | 1.9828 |
| 2 | 6144 | 2560 | 160 | 8 | 3072 | 4.7416 | 1.9796 |
| 3 | 6144 | 2560 | 160 | 8 | 1280 | 3.2593 | 1.9738 |
| 4 | 6144 | 2560 | 160 | 8 | 3089 | 4.6032 | 1.9796 |
| 5 | 6144 | 2560 | 160 | 8 | 1571 | 3.4369 | 1.9747 |
| 6 | 6144 | 2560 | 160 | 8 | 1312 | 3.2878 | 1.9739 |
| 7 | 6144 | 2560 | 160 | 8 | 1344 | 3.2913 | 1.9740 |
| 8 | 6144 | 2560 | 160 | 8 | 1376 | 3.3198 | 1.9741 |
| 9 | 6144 | 2560 | 160 | 8 | 8192 | 9.7553 | 3.4150 |
| 10 | 6144 | 2560 | 160 | 8 | 1408 | 3.3200 | 1.9742 |
| 11 | 6144 | 2560 | 160 | 8 | 2048 | 3.6739 | 1.9763 |
| 12 | 6144 | 2560 | 160 | 8 | 1440 | 3.3385 | 1.9743 |
| 13 | 6144 | 2560 | 160 | 8 | 1536 | 3.4049 | 1.9746 |
| 14 | 6144 | 2560 | 160 | 8 | 1472 | 3.3654 | 1.9744 |
| 15 | 6144 | 2560 | 160 | 8 | 1504 | 3.4042 | 1.9745 |
| 16 | 6144 | 2560 | 160 | 8 | 1568 | 3.4426 | 1.9747 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Generate valid expert indices - each token selects num_experts_per_tok unique experts
    selected_experts = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm

    # Generate routing weights that sum to 1 for each token
    routing_logits = torch.randn(num_tokens, num_experts_per_tok, dtype=dtype, device=device)
    routing_weights = F.softmax(routing_logits.float(), dim=-1).to(dtype)

    # Expert weights: Xavier init with 1/sqrt(fan_in)
    # gate/up: shape [num_experts, hidden_size, moe_intermediate_size], fan_in = hidden_size (dim 1)
    expert_gate_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    expert_up_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    # down: shape [num_experts, moe_intermediate_size, hidden_size], fan_in = moe_intermediate_size (dim 1)
    expert_down_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(moe_intermediate_size)

    return {
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "expert_gate_weights": expert_gate_weights,
        "expert_up_weights": expert_up_weights,
        "expert_down_weights": expert_down_weights,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    capacity = max(int((num_tokens * num_experts_per_tok / num_experts) * 1.25), 1)

    # Flatten all token-expert assignments: (num_tokens * K,)
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)

    # Sort by expert ID (stable to match original sequential assignment order)
    sorted_experts, sorted_indices = flat_experts.sort(stable=True)
    sorted_weights = flat_weights[sorted_indices]
    sorted_token_ids = flat_token_ids[sorted_indices]

    # OPT 1: Vectorized within-expert position computation.
    # After sorting, tokens for the same expert are contiguous.
    # Position within group = global_sorted_index - start_of_expert_group.
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    starts = torch.zeros(num_experts, dtype=torch.long, device=device)
    starts[1:] = counts[:-1].cumsum(0)

    within_pos = torch.arange(len(sorted_experts), device=device) - starts[sorted_experts]

    # Apply capacity constraint (same semantics: first `capacity` per expert admitted)
    valid = within_pos < capacity
    v_exp = sorted_experts[valid]
    v_pos = within_pos[valid]
    v_tok = sorted_token_ids[valid]
    v_wt = sorted_weights[valid]

    # Scatter tokens into padded expert batches using advanced indexing
    expert_inputs = torch.zeros(num_experts, capacity, hidden_size, dtype=dtype, device=device)
    expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

    # Batched expert forward pass (same 3 BMMs as original)
    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)

    # SwiGLU with in-place mul to save one allocation
    activated = F.silu(gate_out).mul_(up_out)

    expert_outputs = torch.bmm(activated, expert_down_weights)

    # OPT 2: Vectorized gather + weighted aggregation.
    # Gather only valid positions (skip mask — invalid positions never read).
    valid_out = expert_outputs[v_exp, v_pos]  # (num_valid, hidden_size)
    weighted_out = v_wt.unsqueeze(1) * valid_out

    result = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    result.index_add_(0, v_tok, weighted_out)

    return result
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 2.044462 ms |
| - | Scoring Baseline | 0.500000 | 3.869265 ms |
| - | Reference Implementation | 0.322690 | 5.836447 ms |
