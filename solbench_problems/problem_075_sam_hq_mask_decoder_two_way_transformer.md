## Description

Two-way transformer block for SAM-HQ mask decoder performing bidirectional cross-attention between sparse prompt tokens and dense image embeddings. Includes: (1) self-attention on sparse tokens, (2) cross-attention from tokens to image, (3) MLP on tokens, (4) cross-attention from image to tokens. Uses downsampled internal dimensions (256->128) for cross-attention efficiency.
| Name | Shape | Dtype |
| --- | --- | --- |
| queries | [batch_size, point_batch_size, n_query_tokens, hidden_size] | float32 |
| keys | [batch_size, point_batch_size, n_key_tokens, hidden_size] | float32 |
| query_point_embedding | [batch_size, point_batch_size, n_query_tokens, hidden_size] | float32 |
| key_point_embedding | [batch_size, point_batch_size, n_key_tokens, hidden_size] | float32 |
| self_attn_q_weight | [hidden_size, hidden_size] | float32 |
| self_attn_q_bias | [hidden_size] | float32 |
| self_attn_k_weight | [hidden_size, hidden_size] | float32 |
| self_attn_k_bias | [hidden_size] | float32 |
| self_attn_v_weight | [hidden_size, hidden_size] | float32 |
| self_attn_v_bias | [hidden_size] | float32 |
| self_attn_out_weight | [hidden_size, hidden_size] | float32 |
| self_attn_out_bias | [hidden_size] | float32 |
| layer_norm1_weight | [hidden_size] | float32 |
| layer_norm1_bias | [hidden_size] | float32 |
| cross_t2i_q_weight | [internal_dim, hidden_size] | float32 |
| cross_t2i_q_bias | [internal_dim] | float32 |
| cross_t2i_k_weight | [internal_dim, hidden_size] | float32 |
| cross_t2i_k_bias | [internal_dim] | float32 |
| cross_t2i_v_weight | [internal_dim, hidden_size] | float32 |
| cross_t2i_v_bias | [internal_dim] | float32 |
| cross_t2i_out_weight | [hidden_size, internal_dim] | float32 |
| cross_t2i_out_bias | [hidden_size] | float32 |
| layer_norm2_weight | [hidden_size] | float32 |
| layer_norm2_bias | [hidden_size] | float32 |
| mlp_lin1_weight | [mlp_dim, hidden_size] | float32 |
| mlp_lin1_bias | [mlp_dim] | float32 |
| mlp_lin2_weight | [hidden_size, mlp_dim] | float32 |
| mlp_lin2_bias | [hidden_size] | float32 |
| layer_norm3_weight | [hidden_size] | float32 |
| layer_norm3_bias | [hidden_size] | float32 |
| cross_i2t_q_weight | [internal_dim, hidden_size] | float32 |
| cross_i2t_q_bias | [internal_dim] | float32 |
| cross_i2t_k_weight | [internal_dim, hidden_size] | float32 |
| cross_i2t_k_bias | [internal_dim] | float32 |
| cross_i2t_v_weight | [internal_dim, hidden_size] | float32 |
| cross_i2t_v_bias | [internal_dim] | float32 |
| cross_i2t_out_weight | [hidden_size, internal_dim] | float32 |
| cross_i2t_out_bias | [hidden_size] | float32 |
| layer_norm4_weight | [hidden_size] | float32 |
| layer_norm4_bias | [hidden_size] | float32 |
| skip_first_layer_pe | scalar | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| output_queries | [batch_size, point_batch_size, n_query_tokens, hidden_size] | float32 |
| output_keys | [batch_size, point_batch_size, n_key_tokens, hidden_size] | float32 |

| # | hidden_size | num_attention_heads | internal_dim | mlp_dim | batch_size | point_batch_size | n_query_tokens | n_key_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | 128 | 2048 | 4 | 2 | 5 | 1024 | 0.4008 | 0.0065 |
| 2 | 256 | 8 | 128 | 2048 | 2 | 1 | 6 | 4096 | 0.4322 | 0.0064 |
| 3 | 256 | 8 | 128 | 2048 | 2 | 1 | 4 | 4096 | 0.4346 | 0.0064 |
| 4 | 256 | 8 | 128 | 2048 | 8 | 1 | 4 | 4096 | 0.7530 | 0.0245 |
| 5 | 256 | 8 | 128 | 2048 | 8 | 1 | 5 | 2048 | 0.5423 | 0.0125 |
| 6 | 256 | 8 | 128 | 2048 | 8 | 2 | 4 | 4096 | 1.1737 | 0.0487 |
| 7 | 256 | 8 | 128 | 2048 | 2 | 2 | 4 | 4096 | 0.5719 | 0.0125 |
| 8 | 256 | 8 | 128 | 2048 | 1 | 1 | 8 | 4096 | 6.8098 | 0.0034 |
| 9 | 256 | 8 | 128 | 2048 | 1 | 4 | 4 | 4096 | 0.6205 | 0.0125 |
| 10 | 256 | 8 | 128 | 2048 | 1 | 2 | 5 | 1024 | 0.3411 | 0.0019 |
| 11 | 256 | 8 | 128 | 2048 | 16 | 1 | 4 | 4096 | 1.1825 | 0.0487 |
| 12 | 256 | 8 | 128 | 2048 | 4 | 1 | 6 | 4096 | 0.5715 | 0.0125 |
| 13 | 256 | 8 | 128 | 2048 | 1 | 1 | 6 | 4096 | 0.7323 | 0.0034 |
| 14 | 256 | 8 | 128 | 2048 | 1 | 1 | 4 | 4096 | 0.7090 | 0.0034 |
| 15 | 256 | 8 | 128 | 2048 | 4 | 1 | 5 | 2048 | 0.4288 | 0.0065 |
| 16 | 256 | 8 | 128 | 2048 | 2 | 2 | 8 | 4096 | 0.8130 | 0.0125 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_point_embedding: torch.Tensor,
    key_point_embedding: torch.Tensor,
    self_attn_q_weight: torch.Tensor,
    self_attn_q_bias: torch.Tensor,
    self_attn_k_weight: torch.Tensor,
    self_attn_k_bias: torch.Tensor,
    self_attn_v_weight: torch.Tensor,
    self_attn_v_bias: torch.Tensor,
    self_attn_out_weight: torch.Tensor,
    self_attn_out_bias: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    cross_t2i_q_weight: torch.Tensor,
    cross_t2i_q_bias: torch.Tensor,
    cross_t2i_k_weight: torch.Tensor,
    cross_t2i_k_bias: torch.Tensor,
    cross_t2i_v_weight: torch.Tensor,
    cross_t2i_v_bias: torch.Tensor,
    cross_t2i_out_weight: torch.Tensor,
    cross_t2i_out_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    mlp_lin1_weight: torch.Tensor,
    mlp_lin1_bias: torch.Tensor,
    mlp_lin2_weight: torch.Tensor,
    mlp_lin2_bias: torch.Tensor,
    layer_norm3_weight: torch.Tensor,
    layer_norm3_bias: torch.Tensor,
    cross_i2t_q_weight: torch.Tensor,
    cross_i2t_q_bias: torch.Tensor,
    cross_i2t_k_weight: torch.Tensor,
    cross_i2t_k_bias: torch.Tensor,
    cross_i2t_v_weight: torch.Tensor,
    cross_i2t_v_bias: torch.Tensor,
    cross_i2t_out_weight: torch.Tensor,
    cross_i2t_out_bias: torch.Tensor,
    layer_norm4_weight: torch.Tensor,
    layer_norm4_bias: torch.Tensor,
    skip_first_layer_pe: bool,
):
    batch_size, point_batch_size, n_query_tokens, hidden_size = queries.shape
    _, _, n_key_tokens, _ = keys.shape
    num_heads = 8
    head_dim_self = hidden_size // num_heads  # 32 for self-attention
    internal_dim = 128
    head_dim_cross = internal_dim // num_heads  # 16 for cross-attention
    eps = 1e-6
    
    def separate_heads(x, num_heads, head_dim):
        # x: (batch, point_batch, n_tokens, channel)
        b, pb, n, c = x.shape
        x = x.reshape(b * pb, n, num_heads, head_dim)
        return x.transpose(1, 2)  # (b*pb, num_heads, n, head_dim)
    
    def recombine_heads(x, point_batch_size):
        # x: (b*pb, num_heads, n, head_dim)
        bpb, nh, n, hd = x.shape
        b = bpb // point_batch_size
        return x.transpose(1, 2).reshape(b, point_batch_size, n, nh * hd)
    
    def attention(q, k, v, num_heads, head_dim, point_batch_size):
        scaling = head_dim ** -0.5
        q = separate_heads(q, num_heads, head_dim)
        k = separate_heads(k, num_heads, head_dim)
        v = separate_heads(v, num_heads, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        return recombine_heads(attn_output, point_batch_size)
    
    def layer_norm(x, weight, bias):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps) * weight + bias
    
    # 1. Self-attention on queries
    if skip_first_layer_pe:
        q_input = queries
    else:
        q_input = queries + query_point_embedding
    
    q_proj = F.linear(q_input, self_attn_q_weight, self_attn_q_bias)
    k_proj = F.linear(q_input, self_attn_k_weight, self_attn_k_bias)
    v_proj = F.linear(queries, self_attn_v_weight, self_attn_v_bias)
    
    attn_out = attention(q_proj, k_proj, v_proj, num_heads, head_dim_self, point_batch_size)
    attn_out = F.linear(attn_out, self_attn_out_weight, self_attn_out_bias)
    queries = queries + attn_out
    queries = layer_norm(queries, layer_norm1_weight, layer_norm1_bias)
    
    # 2. Cross-attention: tokens -> image
    q_input = queries + query_point_embedding
    k_input = keys + key_point_embedding
    
    q_proj = F.linear(q_input, cross_t2i_q_weight, cross_t2i_q_bias)
    k_proj = F.linear(k_input, cross_t2i_k_weight, cross_t2i_k_bias)
    v_proj = F.linear(keys, cross_t2i_v_weight, cross_t2i_v_bias)
    
    attn_out = attention(q_proj, k_proj, v_proj, num_heads, head_dim_cross, point_batch_size)
    attn_out = F.linear(attn_out, cross_t2i_out_weight, cross_t2i_out_bias)
    queries = queries + attn_out
    queries = layer_norm(queries, layer_norm2_weight, layer_norm2_bias)
    
    # 3. MLP block
    mlp_out = F.linear(queries, mlp_lin1_weight, mlp_lin1_bias)
    mlp_out = F.relu(mlp_out)
    mlp_out = F.linear(mlp_out, mlp_lin2_weight, mlp_lin2_bias)
    queries = queries + mlp_out
    queries = layer_norm(queries, layer_norm3_weight, layer_norm3_bias)
    
    # 4. Cross-attention: image -> tokens
    q_input = queries + query_point_embedding
    k_input = keys + key_point_embedding
    
    # Note: q is from keys (image), k is from queries (tokens)
    q_proj = F.linear(k_input, cross_i2t_q_weight, cross_i2t_q_bias)
    k_proj = F.linear(q_input, cross_i2t_k_weight, cross_i2t_k_bias)
    v_proj = F.linear(queries, cross_i2t_v_weight, cross_i2t_v_bias)
    
    attn_out = attention(q_proj, k_proj, v_proj, num_heads, head_dim_cross, point_batch_size)
    attn_out = F.linear(attn_out, cross_i2t_out_weight, cross_i2t_out_bias)
    keys = keys + attn_out
    keys = layer_norm(keys, layer_norm4_weight, layer_norm4_bias)
    
    return queries, keys
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.009120 ms |
| - | Scoring Baseline | 0.500000 | 0.703972 ms |
| - | Reference Implementation | 0.368801 | 1.184275 ms |
