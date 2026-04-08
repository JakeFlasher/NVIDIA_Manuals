## Description

Flux concatenated sequence processing pattern: concatenates text and image sequences, applies linear projection, then splits back. This pattern appears in dual-stream transformer blocks and represents a memory bandwidth bottleneck.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, img_seq_len, hidden_dim] | float32 |
| encoder_hidden_states | [batch_size, text_seq_len, hidden_dim] | float32 |
| process_weight | [hidden_dim, hidden_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| processed_encoder | [batch_size, text_seq_len, hidden_dim] | float32 |
| processed_hidden | [batch_size, img_seq_len, hidden_dim] | float32 |

| # | hidden_dim | batch_size | text_seq_len | img_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 2 | 128 | 256 | 0.0704 | 0.0084 |
| 2 | 3072 | 1 | 256 | 512 | 0.0666 | 0.0084 |
| 3 | 3072 | 2 | 131 | 293 | 0.0730 | 0.0092 |
| 4 | 3072 | 1 | 128 | 256 | 0.0519 | 0.0044 |
| 5 | 3072 | 1 | 77 | 4096 | 0.1579 | 0.0439 |
| 6 | 3072 | 1 | 512 | 2048 | 0.1191 | 0.0271 |
| 7 | 3072 | 1 | 77 | 1024 | 0.0797 | 0.0119 |
| 8 | 3072 | 4 | 77 | 1024 | 0.1682 | 0.0463 |
| 9 | 3072 | 1 | 1024 | 4096 | 0.1844 | 0.0538 |
| 10 | 3072 | 16 | 128 | 256 | 0.2104 | 0.0644 |
| 11 | 3072 | 32 | 256 | 512 | 0.6356 | 0.2565 |
| 12 | 3072 | 4 | 256 | 512 | 0.1352 | 0.0324 |
| 13 | 3072 | 2 | 1423 | 1489 | 0.1888 | 0.0611 |
| 14 | 3072 | 1 | 1087 | 1163 | 0.1134 | 0.0238 |
| 15 | 3072 | 4 | 211 | 449 | 0.1320 | 0.0279 |
| 16 | 3072 | 4 | 128 | 256 | 0.0927 | 0.0164 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flux concatenated sequence processing pattern.
    
    1. Concatenate encoder_hidden_states and hidden_states along sequence dimension
    2. Apply linear projection to the concatenated sequence
    3. Split back into separate encoder and image streams
    
    Args:
        hidden_states: Image latent sequence [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: Text conditioning sequence [batch, text_seq_len, hidden_dim]
        process_weight: Linear projection weight [hidden_dim, hidden_dim]
        
    Returns:
        Tuple of (processed_encoder_hidden_states, processed_hidden_states)
    """
    text_seq_len = encoder_hidden_states.shape[1]
    img_seq_len = hidden_states.shape[1]
    
    # Step 1: Concatenate sequences along sequence dimension
    # Shape: [batch, text_seq_len + img_seq_len, hidden_dim]
    concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    
    # Step 2: Apply linear projection (no bias)
    # processed = concatenated @ process_weight.T
    processed = torch.matmul(concatenated, process_weight.t())
    
    # Step 3: Split back into separate streams
    processed_encoder = processed[:, :text_seq_len, :]
    processed_hidden = processed[:, text_seq_len:, :]
    
    return processed_encoder, processed_hidden
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.025684 ms |
| - | Scoring Baseline | 0.500000 | 0.125755 ms |
| - | Reference Implementation | 0.090834 | 1.101174 ms |
