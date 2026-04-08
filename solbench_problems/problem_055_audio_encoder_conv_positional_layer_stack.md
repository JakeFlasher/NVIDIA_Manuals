## Description

Complete fused audio preprocessing pipeline that performs two strided 1D convolutions with GELU activations on mel spectrogram input, adds learned positional embeddings, and processes through the first encoder layer's self-attention and FFN blocks. This represents the critical path from raw audio features to contextualized audio embeddings.
| Name | Shape | Dtype |
| --- | --- | --- |
| input_features | [batch_size, num_mel_bins, input_seq_len] | bfloat16 |
| conv1_weight | [d_model, num_mel_bins, 3] | bfloat16 |
| conv1_bias | [d_model] | bfloat16 |
| conv2_weight | [d_model, d_model, 3] | bfloat16 |
| conv2_bias | [d_model] | bfloat16 |
| embed_positions_weight | [output_seq_len, d_model] | bfloat16 |
| self_attn_layer_norm_weight | [d_model] | bfloat16 |
| self_attn_layer_norm_bias | [d_model] | bfloat16 |
| q_proj_weight | [d_model, d_model] | bfloat16 |
| q_proj_bias | [d_model] | bfloat16 |
| k_proj_weight | [d_model, d_model] | bfloat16 |
| v_proj_weight | [d_model, d_model] | bfloat16 |
| v_proj_bias | [d_model] | bfloat16 |
| out_proj_weight | [d_model, d_model] | bfloat16 |
| out_proj_bias | [d_model] | bfloat16 |
| final_layer_norm_weight | [d_model] | bfloat16 |
| final_layer_norm_bias | [d_model] | bfloat16 |
| fc1_weight | [encoder_ffn_dim, d_model] | bfloat16 |
| fc1_bias | [encoder_ffn_dim] | bfloat16 |
| fc2_weight | [d_model, encoder_ffn_dim] | bfloat16 |
| fc2_bias | [d_model] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, output_seq_len, d_model] | bfloat16 |

| # | input_seq_len | output_seq_len | num_mel_bins | d_model | num_heads | head_dim | encoder_ffn_dim | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 16 | 18.8608 | 10.8932 |
| 2 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 3 | 3.7081 | 2.0428 |
| 3 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 23 | 27.3361 | 15.6588 |
| 4 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 56 | 67.0146 | 38.1252 |
| 5 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 8 | 9.6171 | 5.4468 |
| 6 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 7 | 8.4324 | 4.7660 |
| 7 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 40 | 49.1232 | 27.2324 |
| 8 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 29 | 35.0550 | 19.7436 |
| 9 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 80 | 98.9470 | 54.4644 |
| 10 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 19 | 22.6722 | 12.9356 |
| 11 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 24 | 28.8832 | 16.3396 |
| 12 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 12 | 14.4499 | 8.1700 |
| 13 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 17 | 20.1300 | 11.5740 |
| 14 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 10 | 12.0727 | 6.8084 |
| 15 | 3000 | 1500 | 80 | 5120 | 20 | 256 | 20480 | 6 | 7.2466 | 4.0852 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    input_seq_len = axes_and_scalars["input_seq_len"]
    output_seq_len = axes_and_scalars["output_seq_len"]
    num_mel_bins = axes_and_scalars["num_mel_bins"]
    d_model = axes_and_scalars["d_model"]
    encoder_ffn_dim = axes_and_scalars["encoder_ffn_dim"]

    dtype = torch.bfloat16

    input_features = torch.randn(batch_size, num_mel_bins, input_seq_len, dtype=dtype, device=device)

    # Conv1 weight: Kaiming init for conv1d (fan_in = in_channels * kernel_size)
    conv1_weight = torch.randn(d_model, num_mel_bins, 3, dtype=dtype, device=device) / math.sqrt(num_mel_bins * 3)
    conv1_bias = torch.zeros(d_model, dtype=dtype, device=device)

    # Conv2 weight: Kaiming init for conv1d (fan_in = in_channels * kernel_size)
    conv2_weight = torch.randn(d_model, d_model, 3, dtype=dtype, device=device) / math.sqrt(d_model * 3)
    conv2_bias = torch.zeros(d_model, dtype=dtype, device=device)

    # Positional embedding: small-magnitude learned embedding init
    embed_positions_weight = torch.randn(output_seq_len, d_model, dtype=dtype, device=device) * 0.02

    # Norm weights/biases
    self_attn_layer_norm_weight = torch.ones(d_model, dtype=dtype, device=device)
    self_attn_layer_norm_bias = torch.zeros(d_model, dtype=dtype, device=device)

    # Projection weights: Xavier init (fan_in = last dim)
    q_proj_weight = torch.randn(d_model, d_model, dtype=dtype, device=device) / math.sqrt(d_model)
    q_proj_bias = torch.zeros(d_model, dtype=dtype, device=device)
    k_proj_weight = torch.randn(d_model, d_model, dtype=dtype, device=device) / math.sqrt(d_model)
    v_proj_weight = torch.randn(d_model, d_model, dtype=dtype, device=device) / math.sqrt(d_model)
    v_proj_bias = torch.zeros(d_model, dtype=dtype, device=device)
    out_proj_weight = torch.randn(d_model, d_model, dtype=dtype, device=device) / math.sqrt(d_model)
    out_proj_bias = torch.zeros(d_model, dtype=dtype, device=device)

    # FFN norm weights/biases
    final_layer_norm_weight = torch.ones(d_model, dtype=dtype, device=device)
    final_layer_norm_bias = torch.zeros(d_model, dtype=dtype, device=device)

    # FFN weights: Xavier init
    fc1_weight = torch.randn(encoder_ffn_dim, d_model, dtype=dtype, device=device) / math.sqrt(d_model)
    fc1_bias = torch.zeros(encoder_ffn_dim, dtype=dtype, device=device)
    fc2_weight = torch.randn(d_model, encoder_ffn_dim, dtype=dtype, device=device) / math.sqrt(encoder_ffn_dim)
    fc2_bias = torch.zeros(d_model, dtype=dtype, device=device)

    return {
        "input_features": input_features,
        "conv1_weight": conv1_weight,
        "conv1_bias": conv1_bias,
        "conv2_weight": conv2_weight,
        "conv2_bias": conv2_bias,
        "embed_positions_weight": embed_positions_weight,
        "self_attn_layer_norm_weight": self_attn_layer_norm_weight,
        "self_attn_layer_norm_bias": self_attn_layer_norm_bias,
        "q_proj_weight": q_proj_weight,
        "q_proj_bias": q_proj_bias,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "v_proj_bias": v_proj_bias,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "final_layer_norm_weight": final_layer_norm_weight,
        "final_layer_norm_bias": final_layer_norm_bias,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
    }

@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    embed_positions_weight: torch.Tensor,
    self_attn_layer_norm_weight: torch.Tensor,
    self_attn_layer_norm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    final_layer_norm_weight: torch.Tensor,
    final_layer_norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    # Constants
    d_model = 5120
    num_heads = 20
    head_dim = 256
    scaling = head_dim ** -0.5
    
    # Stage 1: Convolutional feature extraction
    # Conv1: (B, 80, 3000) -> (B, 5120, 3000)
    x = F.conv1d(input_features, conv1_weight, conv1_bias, padding=1)
    x = F.gelu(x)
    
    # Conv2 with stride=2: (B, 5120, 3000) -> (B, 5120, 1500)
    x = F.conv1d(x, conv2_weight, conv2_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Permute to (B, 1500, 5120)
    x = x.permute(0, 2, 1)
    
    bsz, seq_len, _ = x.shape
    
    # Stage 2: Add positional embeddings
    hidden_states = x + embed_positions_weight
    
    # Stage 3: First encoder layer - Self-attention block
    residual = hidden_states
    
    # Layer norm
    hidden_states_fp32 = hidden_states.to(torch.float32)
    mean = hidden_states_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_states_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states_norm = (hidden_states_fp32 - mean) / torch.sqrt(var + 1e-5)
    hidden_states = (hidden_states_norm * self_attn_layer_norm_weight.to(torch.float32) + self_attn_layer_norm_bias.to(torch.float32)).to(torch.bfloat16)
    
    # Q, K, V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias) * scaling
    key_states = F.linear(hidden_states, k_proj_weight, None)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    
    # Reshape for multi-head attention: (B, seq_len, d_model) -> (B, num_heads, seq_len, head_dim)
    query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    key_states = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    value_states = value_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    
    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, seq_len, d_model)
    
    # Output projection
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    
    # Residual connection
    hidden_states = residual + attn_output
    
    # Stage 4: FFN block
    residual = hidden_states
    
    # Layer norm
    hidden_states_fp32 = hidden_states.to(torch.float32)
    mean = hidden_states_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_states_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states_norm = (hidden_states_fp32 - mean) / torch.sqrt(var + 1e-5)
    hidden_states = (hidden_states_norm * final_layer_norm_weight.to(torch.float32) + final_layer_norm_bias.to(torch.float32)).to(torch.bfloat16)
    
    # FFN
    hidden_states = F.linear(hidden_states, fc1_weight, fc1_bias)
    hidden_states = F.gelu(hidden_states)
    hidden_states = F.linear(hidden_states, fc2_weight, fc2_bias)
    
    # Residual connection
    hidden_states = residual + hidden_states
    
    return hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 11.218917 ms |
| - | Scoring Baseline | 0.500000 | 19.860465 ms |
| - | Reference Implementation | 0.127734 | 70.286670 ms |
