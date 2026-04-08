## Description

Final language model head projection from hidden states to vocabulary logits with optional slicing for generation efficiency. Projects [batch, seq_len, 2048] to [batch, logits_to_keep, 102400]. During generation, logits_to_keep specifies how many trailing positions to compute (typically 1 for autoregressive decoding).
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| weight | [vocab_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| logits | [batch_size, logits_to_keep, vocab_size] | bfloat16 |

| # | hidden_size | vocab_size | batch_size | seq_len | logits_to_keep | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 102400 | 32 | 128 | 128 | 1.0797 | 0.9489 |
| 2 | 2048 | 102400 | 2 | 128 | 128 | 0.1368 | 0.0619 |
| 3 | 2048 | 102400 | 64 | 128 | 128 | 2.1784 | 1.8974 |
| 4 | 2048 | 102400 | 1 | 128 | 128 | 0.1038 | 0.0585 |
| 5 | 2048 | 102400 | 4 | 512 | 512 | 0.5514 | 0.4747 |
| 6 | 2048 | 102400 | 4 | 256 | 256 | 0.3648 | 0.2375 |
| 7 | 2048 | 102400 | 1 | 293 | 293 | 0.1222 | 0.0682 |
| 8 | 2048 | 102400 | 1 | 1024 | 1024 | 0.2968 | 0.2375 |
| 9 | 2048 | 102400 | 4 | 853 | 853 | 0.9668 | 0.7905 |
| 10 | 2048 | 102400 | 1 | 2048 | 2048 | 0.5516 | 0.4747 |
| 11 | 2048 | 102400 | 2 | 2048 | 2048 | 1.0771 | 0.9489 |
| 12 | 2048 | 102400 | 4 | 997 | 997 | 1.0694 | 0.9239 |
| 13 | 2048 | 102400 | 1 | 256 | 256 | 0.1121 | 0.0619 |
| 14 | 2048 | 102400 | 1 | 3011 | 3011 | 0.8051 | 0.6977 |
| 15 | 2048 | 102400 | 1 | 691 | 691 | 0.2739 | 0.1604 |
| 16 | 2048 | 102400 | 8 | 128 | 128 | 0.2968 | 0.2375 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """
    Generate inputs for the LM head projection kernel.
    
    Args:
        axes_and_scalars: Dictionary containing axis values
        device: Target device for tensors
    
    Returns:
        Dictionary of input tensors
    """
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    vocab_size = axes_and_scalars["vocab_size"]
    
    # Generate random hidden states
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    
    # Generate random weight matrix with proper initialization
    std = 1.0 / (hidden_size ** 0.5)
    weight = torch.randn(
        vocab_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * std
    
    return {
        "hidden_states": hidden_states,
        "weight": weight
    }

@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    LM head projection with logit slicing.
    
    This kernel performs:
    1. Slice the last logits_to_keep positions from hidden_states
    2. Project sliced hidden states to vocabulary space via matmul with weight
    
    Args:
        hidden_states: [batch_size, seq_len, 2048] input hidden states
        weight: [102400, 2048] projection weight matrix
    
    Returns:
        logits: [batch_size, logits_to_keep, 102400] vocabulary logits
    
    Note: logits_to_keep is determined by the output shape requirement.
    The kernel slices hidden_states[:, -logits_to_keep:, :] before projection.
    For this reference implementation, we compute all positions and the
    slicing is handled by the benchmark framework based on output shape.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    vocab_size = weight.shape[0]
    
    # Perform the projection: [B, S, H] @ [H, V] = [B, S, V]
    # Using matmul with transposed weight
    # The slicing to logits_to_keep is implicit in the output shape
    logits = torch.matmul(hidden_states, weight.t())
    
    return logits
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.307661 ms |
| 1st place | AKO4ALL_L1 | 0.540017 | 0.396450 ms |
| - | Scoring Baseline | 0.500000 | 0.424157 ms |
| - | Reference Implementation | 0.483986 | 0.426730 ms |
