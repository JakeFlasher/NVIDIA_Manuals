## Description

Critical atomic MoE operation that performs weighted expert output accumulation using index_add_ with element-wise routing weight multiplication. Expert outputs are multiplied by routing weights and scattered back to original token positions using atomic additions.
| Name | Shape | Dtype |
| --- | --- | --- |
| final_hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| expert_outputs | [num_selected_tokens, hidden_size] | bfloat16 |
| token_indices | [num_selected_tokens] | int64 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_experts_per_tok | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 8 | 2 | 1024 | 0.2225 | 0.0037 |
| 2 | 3072 | 8 | 4 | 541 | 0.2342 | 0.0039 |
| 3 | 3072 | 8 | 2 | 128 | 0.0491 | 0.0008 |
| 4 | 3072 | 8 | 1 | 8192 | 0.8360 | 0.0135 |
| 5 | 3072 | 8 | 4 | 512 | 0.2249 | 0.0037 |
| 6 | 3072 | 8 | 1 | 512 | 0.0700 | 0.0012 |
| 7 | 3072 | 8 | 16 | 256 | 0.4252 | 0.0070 |
| 8 | 3072 | 8 | 2 | 512 | 0.1188 | 0.0020 |
| 9 | 3072 | 8 | 4 | 256 | 0.1207 | 0.0020 |
| 10 | 3072 | 8 | 1 | 131 | 0.0402 | 0.0006 |
| 11 | 3072 | 8 | 1 | 1024 | 0.1182 | 0.0020 |
| 12 | 3072 | 8 | 2 | 1879 | 0.3915 | 0.0064 |
| 13 | 3072 | 8 | 2 | 256 | 0.0704 | 0.0012 |
| 14 | 3072 | 8 | 1 | 256 | 0.0457 | 0.0008 |
| 15 | 3072 | 8 | 64 | 128 | 0.8360 | 0.0135 |
| 16 | 3072 | 8 | 32 | 256 | 0.8360 | 0.0135 |

```python
import torch

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Returns the input arguments for the reference forward pass. Required method."""
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    # Initialize accumulation buffer with random values (not zeros) to detect no-op
    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert outputs (weighted outputs from expert computation)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Token indices (which token position each expert output belongs to)
    # These should be in range [0, batch_seq_len)
    # Simulate scattered indices with potential duplicates (for top_k > 1)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }

@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Performs atomic accumulation of expert outputs back to token positions.
    
    Args:
        final_hidden_states: Accumulation buffer for all tokens (batch_seq_len, hidden_size)
        expert_outputs: Weighted outputs from expert computation (num_selected_tokens, hidden_size)
        token_indices: Original token positions (num_selected_tokens,)
        
    Returns:
        Updated final_hidden_states with expert contributions added
    """
    # Clone to avoid modifying input in-place for reference correctness
    output = final_hidden_states.clone()
    
    # Critical atomic accumulation operation
    # This performs: output[token_indices[i]] += expert_outputs[i]
    # for all i in parallel with atomic semantics
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)

    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002907 ms |
| - | Scoring Baseline | 0.500000 | 0.176154 ms |
| - | Reference Implementation | 0.316594 | 0.380065 ms |
