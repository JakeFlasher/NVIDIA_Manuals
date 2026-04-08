## Description

Joint attention context projection that concatenates image and context attention outputs along the feature dimension (duplicating the concatenated sequence) and projects through a linear layer, then splits back into image and context streams. This is a fused operation combining concatenation, linear projection, and sequence splitting.
| Name | Shape | Dtype |
| --- | --- | --- |
| image_attention_output | [batch_size, image_seq_len, inner_dim] | float32 |
| context_attention_output | [batch_size, context_seq_len, inner_dim] | float32 |
| to_out_weight | [inner_dim, proj_input_dim] | float32 |
| to_out_bias | [inner_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| projected_image | [batch_size, image_seq_len, inner_dim] | float32 |
| projected_context | [batch_size, context_seq_len, inner_dim] | float32 |

| # | inner_dim | proj_input_dim | batch_size | image_seq_len | context_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2432 | 4864 | 1 | 1879 | 449 | 0.8331 | 0.0156 |
| 2 | 2432 | 4864 | 4 | 2048 | 512 | 3.0709 | 0.0673 |
| 3 | 2432 | 4864 | 1 | 512 | 128 | 0.2713 | 0.0062 |
| 4 | 2432 | 4864 | 4 | 1024 | 77 | 1.3709 | 0.0292 |
| 5 | 2432 | 4864 | 8 | 1024 | 77 | 2.5781 | 0.0579 |
| 6 | 2432 | 4864 | 8 | 1571 | 154 | 3.6154 | 0.0905 |
| 7 | 2432 | 4864 | 2 | 4096 | 77 | 2.5750 | 0.0549 |
| 8 | 2432 | 4864 | 4 | 2053 | 77 | 2.5700 | 0.0560 |
| 9 | 2432 | 4864 | 2 | 1024 | 77 | 0.8344 | 0.0148 |
| 10 | 2432 | 4864 | 2 | 2048 | 154 | 1.3720 | 0.0292 |
| 11 | 2432 | 4864 | 8 | 512 | 128 | 1.5163 | 0.0338 |
| 12 | 2432 | 4864 | 1 | 8192 | 154 | 2.5691 | 0.0549 |
| 13 | 2432 | 4864 | 1 | 1024 | 77 | 0.4399 | 0.0076 |
| 14 | 2432 | 4864 | 1 | 4096 | 77 | 1.3656 | 0.0277 |
| 15 | 2432 | 4864 | 64 | 128 | 77 | 3.6042 | 0.0861 |
| 16 | 2432 | 4864 | 32 | 256 | 77 | 3.0848 | 0.0700 |

```python
import torch

@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    """
    Joint attention context projection.
    
    1. Concatenate image and context attention outputs along sequence dimension
    2. Duplicate along feature dimension (creating [features, features] pattern)
    3. Project through linear layer
    4. Split back into image and context streams
    
    Args:
        image_attention_output: (batch_size, image_seq_len, inner_dim)
        context_attention_output: (batch_size, context_seq_len, inner_dim)
        to_out_weight: (inner_dim, 2*inner_dim)
        to_out_bias: (inner_dim,)
    
    Returns:
        projected_image: (batch_size, image_seq_len, inner_dim)
        projected_context: (batch_size, context_seq_len, inner_dim)
    """
    batch_size = image_attention_output.shape[0]
    image_seq_len = image_attention_output.shape[1]
    context_seq_len = context_attention_output.shape[1]
    
    # Concatenate image and context attention outputs along sequence dimension
    # Shape: (batch_size, image_seq_len + context_seq_len, inner_dim)
    combined_attention = torch.cat(
        [image_attention_output, context_attention_output],
        dim=1
    )
    
    # Duplicate along feature dimension for joint processing
    # Shape: (batch_size, image_seq_len + context_seq_len, 2 * inner_dim)
    combined_features = torch.cat(
        [combined_attention, combined_attention],
        dim=-1
    )
    
    # Project through output layer: linear(x) = x @ W^T + b
    # Shape: (batch_size, image_seq_len + context_seq_len, inner_dim)
    projected_output = torch.matmul(combined_features, to_out_weight.t()) + to_out_bias
    
    # Split back into image and context streams
    projected_image = projected_output[:, :image_seq_len, :]
    projected_context = projected_output[:, image_seq_len:, :]
    
    return projected_image, projected_context
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.033950 ms |
| - | Scoring Baseline | 0.500000 | 1.594611 ms |
| - | Reference Implementation | 0.356173 | 2.856358 ms |
