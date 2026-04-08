## Description

Complete decoder layer with pre-normalization, self-attention (GQA with Q/K norm), MoE MLP, and residual connections. Includes RMSNorm before attention and MLP, rotary position embeddings, and sparse mixture of experts routing.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | float32 |
| input_layernorm_weight | [hidden_size] | float32 |
| q_proj_weight | [q_out_features, hidden_size] | float32 |
| q_proj_bias | [q_out_features] | float32 |
| k_proj_weight | [kv_out_features, hidden_size] | float32 |
| k_proj_bias | [kv_out_features] | float32 |
| v_proj_weight | [kv_out_features, hidden_size] | float32 |
| v_proj_bias | [kv_out_features] | float32 |
| q_norm_weight | [head_dim] | float32 |
| k_norm_weight | [head_dim] | float32 |
| o_proj_weight | [hidden_size, q_out_features] | float32 |
| o_proj_bias | [hidden_size] | float32 |
| post_attention_layernorm_weight | [hidden_size] | float32 |
| router_weight | [num_experts, hidden_size] | float32 |
| expert_gate_weights | [num_experts, moe_intermediate_size, hidden_size] | float32 |
| expert_up_weights | [num_experts, moe_intermediate_size, hidden_size] | float32 |
| expert_down_weights | [num_experts, hidden_size, moe_intermediate_size] | float32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | moe_intermediate_size | num_experts | num_experts_per_tok | q_out_features | kv_out_features | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 8 | 373 | 2.4026 | 0.1979 |
| 2 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 16 | 449 | 4.8568 | 0.4808 |
| 3 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 1056 | 1.7223 | 0.1637 |
| 4 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 8192 | 17.1273 | 1.1220 |
| 5 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 2 | 544 | 1.6263 | 0.1638 |
| 6 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 8 | 128 | 1.4423 | 0.1635 |
| 7 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 1024 | 1.6751 | 0.1635 |
| 8 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 2 | 2048 | 4.4267 | 0.3336 |
| 9 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 16 | 128 | 1.8591 | 0.1690 |
| 10 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 32 | 256 | 4.8026 | 0.5339 |
| 11 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 2 | 512 | 1.5554 | 0.1635 |
| 12 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 4096 | 6.1619 | 0.4094 |
| 13 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 1088 | 1.7517 | 0.1638 |
| 14 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 1 | 1120 | 1.7669 | 0.1640 |
| 15 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 4 | 256 | 1.4846 | 0.1635 |
| 16 | 2048 | 32 | 4 | 128 | 768 | 128 | 8 | 4096 | 512 | 8 | 256 | 2.0097 | 0.1690 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 2048
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    moe_intermediate_size = 768
    num_experts = 128
    
    q_out_features = num_attention_heads * head_dim
    kv_out_features = num_key_value_heads * head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    cos = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    sin = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=torch.float32)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = attention_mask.masked_fill(causal_mask, float("-inf"))
    
    input_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    q_proj_weight = torch.randn(q_out_features, hidden_size, device=device, dtype=torch.float32) * 0.02
    q_proj_bias = torch.randn(q_out_features, device=device, dtype=torch.float32) * 0.01
    k_proj_weight = torch.randn(kv_out_features, hidden_size, device=device, dtype=torch.float32) * 0.02
    k_proj_bias = torch.randn(kv_out_features, device=device, dtype=torch.float32) * 0.01
    v_proj_weight = torch.randn(kv_out_features, hidden_size, device=device, dtype=torch.float32) * 0.02
    v_proj_bias = torch.randn(kv_out_features, device=device, dtype=torch.float32) * 0.01
    q_norm_weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    o_proj_weight = torch.randn(hidden_size, q_out_features, device=device, dtype=torch.float32) * 0.02
    o_proj_bias = torch.randn(hidden_size, device=device, dtype=torch.float32) * 0.01
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    router_weight = torch.randn(num_experts, hidden_size, device=device, dtype=torch.float32) * 0.02
    expert_gate_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    expert_up_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    expert_down_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, device=device, dtype=torch.float32) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "cos": cos,
        "sin": sin,
        "attention_mask": attention_mask,
        "input_layernorm_weight": input_layernorm_weight,
        "q_proj_weight": q_proj_weight,
        "q_proj_bias": q_proj_bias,
        "k_proj_weight": k_proj_weight,
        "k_proj_bias": k_proj_bias,
        "v_proj_weight": v_proj_weight,
        "v_proj_bias": v_proj_bias,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "o_proj_weight": o_proj_weight,
        "o_proj_bias": o_proj_bias,
        "post_attention_layernorm_weight": post_attention_layernorm_weight,
        "router_weight": router_weight,
        "expert_gate_weights": expert_gate_weights,
        "expert_up_weights": expert_up_weights,
        "expert_down_weights": expert_down_weights,
        "rms_norm_eps": 1e-6
    }

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    num_experts = 128
    top_k = 8
    
    # First residual connection: around attention
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)
    
    # Q, K, V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Q/K normalization
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply rotary embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat KV for GQA
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, o_proj_weight, o_proj_bias)
    
    hidden_states = residual + attn_output
    
    # Second residual connection: around MLP (MoE)
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, rms_norm_eps)
    
    # MoE routing
    hidden_states_flat = hidden_states.view(-1, hidden_size)
    router_logits = F.linear(hidden_states_flat, router_weight)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
    routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)
    
    # Expert computation
    final_hidden_states = torch.zeros_like(hidden_states_flat)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    for expert_idx in range(num_experts):
        if expert_mask[expert_idx].sum() > 0:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_flat[top_x]
            
            gate_out = F.linear(current_state, expert_gate_weights[expert_idx])
            up_out = F.linear(current_state, expert_up_weights[expert_idx])
            expert_out = F.silu(gate_out) * up_out
            expert_out = F.linear(expert_out, expert_down_weights[expert_idx])
            
            current_hidden_states = expert_out * routing_weights_topk[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
    
    final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
    output = residual + final_hidden_states
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.239150 ms |
| - | Scoring Baseline | 0.500000 | 2.613271 ms |
| - | Reference Implementation | 0.096086 | 24.651488 ms |
