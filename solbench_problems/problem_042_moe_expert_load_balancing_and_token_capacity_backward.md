## Description

Backward pass for MoE expert load balancing and token capacity. Computes gradients through load_balance_loss, sum reduction, and scatter operations. Since topk_idx contains discrete indices and hidden_states is only used for shape, gradients w.r.t. both inputs are None.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_tokens_per_expert | [n_routed_experts] | float32 |
| grad_expert_mask | [batch_seq_len, n_routed_experts] | float32 |
| grad_load_balance_loss | [1] | float32 |
| topk_idx | [batch_seq_len, num_experts_per_tok] | int64 |
| expert_mask | [batch_seq_len, n_routed_experts] | int64 |
| tokens_per_expert | [n_routed_experts] | int32 |
| training | [1] | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_expert_mask_out | [batch_seq_len, n_routed_experts] | float32 |

| # | n_routed_experts | num_experts_per_tok | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | 2371 | 0.0215 | 0.0007 |
| 2 | 256 | 8 | 3072 | 0.0205 | 0.0008 |
| 3 | 256 | 8 | 2048 | 0.0204 | 0.0007 |
| 4 | 256 | 8 | 65536 | 0.0698 | 0.0092 |
| 5 | 256 | 8 | 524288 | 0.1990 | 0.0704 |
| 6 | 256 | 8 | 262144 | 0.1214 | 0.0354 |
| 7 | 256 | 8 | 2080 | 0.0235 | 0.0007 |
| 8 | 256 | 8 | 3557 | 0.0234 | 0.0009 |
| 9 | 256 | 8 | 2112 | 0.0231 | 0.0007 |
| 10 | 256 | 8 | 131072 | 0.0832 | 0.0179 |
| 11 | 256 | 8 | 32768 | 0.0454 | 0.0048 |
| 12 | 256 | 8 | 2144 | 0.0232 | 0.0007 |
| 13 | 256 | 8 | 4096 | 0.0233 | 0.0009 |
| 14 | 256 | 8 | 2176 | 0.0235 | 0.0007 |
| 15 | 256 | 8 | 2208 | 0.0235 | 0.0007 |
| 16 | 256 | 8 | 3169 | 0.0235 | 0.0008 |

```python
import torch

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    n_routed_experts = axes_and_scalars["n_routed_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Gradient inputs
    grad_tokens_per_expert = torch.randn(n_routed_experts, dtype=torch.float32, device=device)
    grad_expert_mask = torch.randn(batch_seq_len, n_routed_experts, dtype=torch.float32, device=device)
    grad_load_balance_loss = torch.randn(1, dtype=torch.float32, device=device)
    
    # Saved tensors from forward pass
    topk_idx = torch.randint(0, n_routed_experts, (batch_seq_len, num_experts_per_tok), dtype=torch.int64, device=device)
    
    # Create expert_mask from topk_idx (as done in forward)
    expert_mask = torch.zeros((batch_seq_len, n_routed_experts), dtype=torch.int64, device=device)
    expert_mask.scatter_(1, topk_idx, 1)
    
    # Compute tokens_per_expert from expert_mask
    tokens_per_expert = expert_mask.sum(dim=0).to(torch.int32)
    
    # Training flag as 1-element bool tensor
    training = torch.tensor([True], dtype=torch.bool, device=device)
    
    return {
        "grad_tokens_per_expert": grad_tokens_per_expert,
        "grad_expert_mask": grad_expert_mask,
        "grad_load_balance_loss": grad_load_balance_loss,
        "topk_idx": topk_idx,
        "expert_mask": expert_mask,
        "tokens_per_expert": tokens_per_expert,
        "training": training,
    }

@torch.no_grad()
def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    """
    Backward pass for MoE expert load balancing.
    
    Computes gradients through:
    1. load_balance_loss computation
    2. sum reduction (tokens_per_expert = expert_mask.sum(dim=0))
    3. scatter operation (discrete, no gradient for topk_idx)
    
    Returns accumulated gradient for expert_mask.
    """
    batch_seq_len = topk_idx.shape[0]
    n_routed_experts = 256
    num_experts_per_tok = 8
    
    # Extract scalar values from tensors
    grad_loss_val = grad_load_balance_loss.item()
    is_training = training.item()
    
    # Initialize output gradient
    grad_expert_mask_out = grad_expert_mask.clone()
    
    # Gradient from load_balance_loss
    # load_balance_loss = n_routed_experts * sum(expert_fraction * uniform_prob)
    # where expert_fraction = tokens_per_expert / (batch_seq_len * num_experts_per_tok)
    # and uniform_prob = 1.0 / n_routed_experts
    #
    # d(load_balance_loss)/d(tokens_per_expert) = n_routed_experts * uniform_prob / (batch_seq_len * num_experts_per_tok)
    #                                            = 1 / (batch_seq_len * num_experts_per_tok)
    
    grad_tpe_accumulated = grad_tokens_per_expert.clone()
    
    if is_training:
        uniform_prob = 1.0 / n_routed_experts
        grad_from_loss = grad_loss_val * n_routed_experts * uniform_prob / (
            batch_seq_len * num_experts_per_tok
        )
        grad_tpe_accumulated = grad_tpe_accumulated + grad_from_loss
    
    # Gradient through sum operation
    # tokens_per_expert = expert_mask.sum(dim=0)
    # d(loss)/d(expert_mask) = d(loss)/d(tokens_per_expert).unsqueeze(0).expand_as(expert_mask)
    
    # Broadcast gradient from [n_routed_experts] to [batch_seq_len, n_routed_experts]
    grad_expert_mask_from_sum = grad_tpe_accumulated.unsqueeze(0).expand(
        batch_seq_len, n_routed_experts
    ).float()
    
    grad_expert_mask_out = grad_expert_mask_out + grad_expert_mask_from_sum
    
    return grad_expert_mask_out

if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={
            "batch_seq_len": 512,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        },
        device=torch.device("cuda:0"),
    )
    out = run(**inputs)
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002024 ms |
| - | Scoring Baseline | 0.500000 | 0.034993 ms |
| - | Reference Implementation | 0.244439 | 0.101987 ms |
