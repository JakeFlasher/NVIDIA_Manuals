## Description

Attention softmax with logit softcapping (tanh-based clamping to ±30.0) applied before softmax, as used in Gemma-2 models. This operation applies: tanh(logits / 30.0) * 30.0 to stabilize attention scores, then softmax normalization.
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_weights | [batch_size, num_heads, seq_len_q, seq_len_k] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, num_heads, seq_len_q, seq_len_k] | bfloat16 |

| # | num_heads | attn_logit_softcapping | batch_size | seq_len_q | seq_len_k | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 30 | 1 | 691 | 691 | 0.0608 | 0.0024 |
| 2 | 8 | 30 | 4 | 512 | 512 | 0.0674 | 0.0048 |
| 3 | 8 | 30 | 2 | 2048 | 2048 | 0.1907 | 0.0354 |
| 4 | 8 | 30 | 4 | 1 | 853 | 0.0363 | 0.0004 |
| 5 | 8 | 30 | 64 | 128 | 128 | 0.0674 | 0.0048 |
| 6 | 8 | 30 | 4 | 256 | 256 | 0.0480 | 0.0015 |
| 7 | 8 | 30 | 16 | 256 | 256 | 0.0530 | 0.0048 |
| 8 | 8 | 30 | 8 | 256 | 256 | 0.0613 | 0.0026 |
| 9 | 8 | 30 | 1 | 293 | 293 | 0.0366 | 0.0008 |
| 10 | 8 | 30 | 8 | 128 | 128 | 0.0368 | 0.0009 |
| 11 | 8 | 30 | 1 | 2048 | 2048 | 0.1156 | 0.0179 |
| 12 | 8 | 30 | 32 | 128 | 128 | 0.0621 | 0.0026 |
| 13 | 8 | 30 | 1 | 512 | 512 | 0.0478 | 0.0015 |
| 14 | 8 | 30 | 2 | 1024 | 1024 | 0.0582 | 0.0092 |
| 15 | 8 | 30 | 4 | 1024 | 1024 | 0.0797 | 0.0179 |
| 16 | 8 | 30 | 8 | 512 | 512 | 0.0579 | 0.0092 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Apply Gemma3's softcapping transformation followed by softmax.
    
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30]
    
    Args:
        attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        
    Returns:
        Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    SOFTCAP = 30.0
    
    # Apply softcapping transformation
    # Step 1: Divide by softcap
    scaled = attn_weights / SOFTCAP
    
    # Step 2: Apply tanh to clamp to [-1, 1]
    clamped = torch.tanh(scaled)
    
    # Step 3: Multiply by softcap to restore scale (now in [-30, 30])
    softcapped = clamped * SOFTCAP
    
    # Apply softmax normalization along the key dimension
    # Upcast to float32 for numerical stability, then cast back
    output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003677 ms |
| 1st place | gnehiz | 0.602574 | 0.039796 ms |
| - | Scoring Baseline | 0.500000 | 0.061013 ms |
| - | Reference Implementation | 0.381199 | 0.097817 ms |
