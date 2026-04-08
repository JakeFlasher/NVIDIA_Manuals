## Description

RWKV's time decay exponential computation with numerical stabilization for the WKV attention mechanism. Computes -exp(time_decay) and manages exponential stability through max-state tracking for linear-time attention.
| Name | Shape | Dtype |
| --- | --- | --- |
| time_decay | [attention_hidden_size] | float32 |
| key | [batch_size, seq_len, attention_hidden_size] | float32 |
| time_first | [attention_hidden_size] | float32 |
| value | [batch_size, seq_len, attention_hidden_size] | float32 |
| max_state | [batch_size, attention_hidden_size] | float32 |
| num_state | [batch_size, attention_hidden_size] | float32 |
| den_state | [batch_size, attention_hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, attention_hidden_size] | float32 |
| max_state_out | [batch_size, attention_hidden_size] | float32 |
| num_state_out | [batch_size, attention_hidden_size] | float32 |
| den_state_out | [batch_size, attention_hidden_size] | float32 |

| # | attention_hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 16 | 2131 | 48.6062 | 0.0915 |
| 2 | 2048 | 4 | 1973 | 46.4873 | 0.0215 |
| 3 | 2048 | 1 | 131 | 3.2112 | 0.0008 |
| 4 | 2048 | 1 | 128 | 3.1183 | 0.0007 |
| 5 | 2048 | 32 | 4096 | 98.6991 | 0.2105 |
| 6 | 2048 | 64 | 613 | 15.0563 | 0.0634 |
| 7 | 2048 | 64 | 1721 | 42.2188 | 0.1770 |
| 8 | 2048 | 8 | 1489 | 35.9292 | 0.0195 |
| 9 | 2048 | 1 | 691 | 16.7868 | 0.0015 |
| 10 | 2048 | 4 | 512 | 12.1029 | 0.0037 |
| 11 | 2048 | 2 | 256 | 6.2449 | 0.0012 |
| 12 | 2048 | 2 | 1879 | 45.1513 | 0.0064 |
| 13 | 2048 | 32 | 541 | 13.2287 | 0.0282 |
| 14 | 2048 | 16 | 449 | 10.5106 | 0.0119 |
| 15 | 2048 | 8 | 1024 | 24.6917 | 0.0135 |
| 16 | 2048 | 8 | 919 | 22.2681 | 0.0122 |

```python
import torch

def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with proper initialization for numerical stability."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    attention_hidden_size = axes_and_scalars["attention_hidden_size"]
    
    # time_decay should be reasonable values (not too large to avoid overflow)
    time_decay = torch.randn(attention_hidden_size, dtype=torch.float32, device=device) * 0.5
    
    # key and value are standard attention tensors
    key = torch.randn(batch_size, seq_len, attention_hidden_size, dtype=torch.float32, device=device)
    value = torch.randn(batch_size, seq_len, attention_hidden_size, dtype=torch.float32, device=device)
    
    # time_first should be reasonable values
    time_first = torch.randn(attention_hidden_size, dtype=torch.float32, device=device) * 0.5
    
    # Initialize max_state to very negative values (like -1e38) for proper initialization
    max_state = torch.full((batch_size, attention_hidden_size), -1e38, dtype=torch.float32, device=device)
    
    # Initialize num_state and den_state to zeros
    num_state = torch.zeros(batch_size, attention_hidden_size, dtype=torch.float32, device=device)
    den_state = torch.zeros(batch_size, attention_hidden_size, dtype=torch.float32, device=device)
    
    return {
        "time_decay": time_decay,
        "key": key,
        "time_first": time_first,
        "value": value,
        "max_state": max_state,
        "num_state": num_state,
        "den_state": den_state,
    }

@torch.no_grad()
def run(
    time_decay: torch.Tensor,
    key: torch.Tensor,
    time_first: torch.Tensor,
    value: torch.Tensor,
    max_state: torch.Tensor,
    num_state: torch.Tensor,
    den_state: torch.Tensor,
):
    """
    RWKV Time Decay Exponential with Numerical Stabilization.
    
    Computes:
    1. time_decay_exp = -exp(time_decay)
    2. For each timestep: max_state = max(max_state + time_decay_exp, current_key)
    3. Exponential normalization: e1 = exp(old_max - new_max), e2 = exp(key - new_max)
    
    This prevents numerical overflow/underflow in the exponential computations
    while maintaining the recurrent state updates.
    """
    batch_size, seq_len, hidden_size = key.size()
    
    # Clone states to avoid modifying inputs
    max_state = max_state.clone().float()
    num_state = num_state.clone().float()
    den_state = den_state.clone().float()
    
    # Compute exponential decay (negative to ensure decay)
    time_decay_exp = -torch.exp(time_decay.float())  # [attention_hidden_size]
    
    output = torch.zeros_like(key, dtype=torch.float32)
    
    # Process each timestep with exponential stabilization
    for t in range(seq_len):
        current_key = key[:, t].float()  # [batch_size, attention_hidden_size]
        current_value = value[:, t].float()  # [batch_size, attention_hidden_size]
        
        # === Output computation with stabilization ===
        # max_for_output prevents overflow in exp(max_state) and exp(current_key + time_first)
        max_for_output = torch.maximum(
            max_state, current_key + time_first
        )  # [batch_size, attention_hidden_size]
        
        # Compute normalized exponentials
        e1_output = torch.exp(max_state - max_for_output)
        e2_output = torch.exp(current_key + time_first - max_for_output)
        
        # WKV computation: weighted sum of values
        numerator = e1_output * num_state + e2_output * current_value
        denominator = e1_output * den_state + e2_output
        output[:, t] = numerator / denominator
        
        # === State update with stabilization ===
        # max_for_state prevents overflow in exp(max_state + time_decay_exp) and exp(current_key)
        max_for_state = torch.maximum(
            max_state + time_decay_exp, current_key
        )  # [batch_size, attention_hidden_size]
        
        # Compute normalized exponentials for state update
        e1_state = torch.exp(max_state + time_decay_exp - max_for_state)
        e2_state = torch.exp(current_key - max_for_state)
        
        # Update recurrent states
        num_state = e1_state * num_state + e2_state * current_value
        den_state = e1_state * den_state + e2_state
        max_state = max_for_state
    
    return output, max_state, num_state, den_state
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.011677 ms |
| - | Scoring Baseline | 0.500000 | 18.698813 ms |
| - | Reference Implementation | 0.167835 | 92.646980 ms |
