## Description

AdaLayerNormZero modulation parameter extraction for Flux dual-stream transformer. Projects timestep embeddings through a linear layer and chunks into 6 modulation parameters (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) that control attention and MLP paths.
| Name | Shape | Dtype |
| --- | --- | --- |
| emb | [batch_size, inner_dim] | float32 |
| weight | [output_dim, inner_dim] | float32 |
| bias | [output_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| shift_msa | [batch_size, inner_dim] | float32 |
| scale_msa | [batch_size, inner_dim] | float32 |
| gate_msa | [batch_size, inner_dim] | float32 |
| shift_mlp | [batch_size, inner_dim] | float32 |
| scale_mlp | [batch_size, inner_dim] | float32 |
| gate_mlp | [batch_size, inner_dim] | float32 |

| # | inner_dim | output_dim | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 18432 | 512 | 0.1339 | 0.0324 |
| 2 | 3072 | 18432 | 211 | 0.1228 | 0.0162 |
| 3 | 3072 | 18432 | 449 | 0.1435 | 0.0285 |
| 4 | 3072 | 18432 | 373 | 0.1208 | 0.0237 |
| 5 | 3072 | 18432 | 16 | 0.0907 | 0.0152 |
| 6 | 3072 | 18432 | 192 | 0.1059 | 0.0161 |
| 7 | 3072 | 18432 | 131 | 0.1019 | 0.0158 |
| 8 | 3072 | 18432 | 919 | 0.1835 | 0.0579 |
| 9 | 3072 | 18432 | 32 | 0.0935 | 0.0153 |
| 10 | 3072 | 18432 | 853 | 0.1772 | 0.0537 |
| 11 | 3072 | 18432 | 128 | 0.0985 | 0.0158 |
| 12 | 3072 | 18432 | 773 | 0.1639 | 0.0487 |
| 13 | 3072 | 18432 | 5 | 0.0902 | 0.0152 |
| 14 | 3072 | 18432 | 384 | 0.1368 | 0.0244 |
| 15 | 3072 | 18432 | 691 | 0.1754 | 0.0436 |
| 16 | 3072 | 18432 | 96 | 0.1439 | 0.0156 |

```python
import torch

@torch.no_grad()
def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    AdaLayerNormZero modulation parameter extraction.
    
    Performs a linear projection followed by chunking into 6 modulation parameters.
    
    Args:
        emb: Timestep embeddings of shape [batch_size, inner_dim]
        weight: Linear weight matrix of shape [6 * inner_dim, inner_dim]
        bias: Linear bias of shape [6 * inner_dim]
        
    Returns:
        Tuple of 6 tensors, each of shape [batch_size, inner_dim]:
        - shift_msa, scale_msa, gate_msa: for attention path
        - shift_mlp, scale_mlp, gate_mlp: for MLP path
    """
    # Linear projection: [batch_size, inner_dim] @ [inner_dim, 6*inner_dim] + bias
    # = [batch_size, 6*inner_dim]
    emb_out = torch.matmul(emb, weight.t()) + bias
    
    # Chunk into 6 equal parts along the last dimension
    # Each chunk has shape [batch_size, inner_dim]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=1)
    
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.024066 ms |
| - | Scoring Baseline | 0.500000 | 0.126430 ms |
| - | Reference Implementation | 0.163585 | 0.685459 ms |
