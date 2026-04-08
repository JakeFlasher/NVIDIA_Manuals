## Description

Collapses multiple AltUp prediction streams back into a single hidden state by projecting each stream through learned unembed matrices, normalizing their magnitudes to match the first stream, and averaging them together.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [altup_num_inputs, batch_size, seq_len, hidden_size] | bfloat16 |
| unembed_proj_1 | [hidden_size, hidden_size] | bfloat16 |
| unembed_proj_2 | [hidden_size, hidden_size] | bfloat16 |
| epsilon | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | altup_num_inputs | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 3 | 1 | 613 | 0.2365 | 0.0132 |
| 2 | 3072 | 3 | 8 | 1024 | 0.5216 | 0.1711 |
| 3 | 3072 | 3 | 16 | 449 | 0.4805 | 0.1501 |
| 4 | 3072 | 3 | 1 | 131 | 0.2200 | 0.0057 |
| 5 | 3072 | 3 | 16 | 256 | 0.3807 | 0.0858 |
| 6 | 3072 | 3 | 1 | 256 | 0.2225 | 0.0061 |
| 7 | 3072 | 3 | 16 | 128 | 0.2970 | 0.0431 |
| 8 | 3072 | 3 | 64 | 256 | 0.7808 | 0.3419 |
| 9 | 3072 | 3 | 64 | 128 | 0.5285 | 0.1711 |
| 10 | 3072 | 3 | 2 | 211 | 0.2287 | 0.0092 |
| 11 | 3072 | 3 | 8 | 373 | 0.3275 | 0.0626 |
| 12 | 3072 | 3 | 16 | 512 | 0.5281 | 0.1711 |
| 13 | 3072 | 3 | 1 | 128 | 0.2308 | 0.0057 |
| 14 | 3072 | 3 | 2 | 128 | 0.2246 | 0.0061 |
| 15 | 3072 | 3 | 32 | 128 | 0.3844 | 0.0858 |
| 16 | 3072 | 3 | 32 | 541 | 0.8071 | 0.3612 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    """
    Collapses multiple AltUp prediction streams into a single hidden state.
    
    Args:
        hidden_states: [altup_num_inputs, batch_size, seq_len, hidden_size]
        unembed_proj_1: [hidden_size, hidden_size] - projection for stream 1
        unembed_proj_2: [hidden_size, hidden_size] - projection for stream 2
        epsilon: small value for numerical stability
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Extract first stream as reference (no projection needed)
    first_stream = hidden_states[0]  # [batch, seq_len, hidden_size]
    
    # Compute target magnitude from first stream
    # Shape: [batch, seq_len, 1]
    target_magnitude = torch.sqrt(
        torch.mean(first_stream.to(torch.float32) ** 2, dim=-1, keepdim=True)
    )
    
    # Initialize list to collect all streams
    collapsed_streams = [first_stream.to(torch.float32)]
    
    # Process stream 1
    # Project through unembed matrix: [batch, seq_len, hidden_size] @ [hidden_size, hidden_size].T
    projected_stream_1 = torch.matmul(
        hidden_states[1].to(torch.float32),
        unembed_proj_1.to(torch.float32).t()
    )
    
    # Compute current magnitude
    current_magnitude_1 = torch.sqrt(
        torch.maximum(
            torch.mean(projected_stream_1 ** 2, dim=-1, keepdim=True),
            torch.tensor(epsilon, dtype=torch.float32, device=hidden_states.device)
        )
    )
    
    # Normalize to match target magnitude
    normalized_stream_1 = projected_stream_1 * (target_magnitude / current_magnitude_1)
    collapsed_streams.append(normalized_stream_1)
    
    # Process stream 2
    projected_stream_2 = torch.matmul(
        hidden_states[2].to(torch.float32),
        unembed_proj_2.to(torch.float32).t()
    )
    
    # Compute current magnitude
    current_magnitude_2 = torch.sqrt(
        torch.maximum(
            torch.mean(projected_stream_2 ** 2, dim=-1, keepdim=True),
            torch.tensor(epsilon, dtype=torch.float32, device=hidden_states.device)
        )
    )
    
    # Normalize to match target magnitude
    normalized_stream_2 = projected_stream_2 * (target_magnitude / current_magnitude_2)
    collapsed_streams.append(normalized_stream_2)
    
    # Stack all streams and compute mean
    stacked_streams = torch.stack(collapsed_streams, dim=0)  # [3, batch, seq_len, hidden_size]
    
    # Average across streams (dim=0)
    output = torch.mean(stacked_streams, dim=0)  # [batch, seq_len, hidden_size]
    
    return output.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.044878 ms |
| - | Scoring Baseline | 0.500000 | 0.361770 ms |
| - | Reference Implementation | 0.287576 | 0.802117 ms |
