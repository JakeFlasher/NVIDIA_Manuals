## Description

Complete end-to-end multimodal fusion pipeline that processes raw audio features through a CTC encoder, applies windowed Q-Former downsampling with learnable queries, and scatters the resulting audio embeddings into language model token embeddings at audio token positions. This is the critical bottleneck that bridges audio and language modalities.
| Name | Shape | Dtype |
| --- | --- | --- |
| input_features | [batch_size, audio_seq_len, encoder_input_dim] | bfloat16 |
| lm_embeddings | [batch_size, text_seq_len, text_hidden_size] | bfloat16 |
| audio_token_positions | [batch_size, num_audio_tokens] | int64 |
| encoder_input_weight | [encoder_hidden_dim, encoder_input_dim] | bfloat16 |
| encoder_input_bias | [encoder_hidden_dim] | bfloat16 |
| encoder_out_weight | [encoder_output_dim, encoder_hidden_dim] | bfloat16 |
| encoder_out_bias | [encoder_output_dim] | bfloat16 |
| encoder_out_mid_weight | [encoder_hidden_dim, encoder_output_dim] | bfloat16 |
| encoder_out_mid_bias | [encoder_hidden_dim] | bfloat16 |
| learnable_queries | [1, num_queries, projector_hidden_size] | bfloat16 |
| qformer_q_proj_weight | [projector_hidden_size, projector_hidden_size] | bfloat16 |
| qformer_q_proj_bias | [projector_hidden_size] | bfloat16 |
| qformer_k_proj_weight | [projector_hidden_size, encoder_hidden_dim] | bfloat16 |
| qformer_k_proj_bias | [projector_hidden_size] | bfloat16 |
| qformer_v_proj_weight | [projector_hidden_size, encoder_hidden_dim] | bfloat16 |
| qformer_v_proj_bias | [projector_hidden_size] | bfloat16 |
| qformer_out_proj_weight | [projector_hidden_size, projector_hidden_size] | bfloat16 |
| qformer_out_proj_bias | [projector_hidden_size] | bfloat16 |
| projector_weight | [text_hidden_size, projector_hidden_size] | bfloat16 |
| projector_bias | [text_hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| fused_embeddings | [batch_size, text_seq_len, text_hidden_size] | bfloat16 |

| # | encoder_input_dim | encoder_hidden_dim | encoder_output_dim | projector_hidden_size | window_size | num_queries | text_hidden_size | qformer_num_heads | qformer_head_dim | batch_size | audio_seq_len | text_seq_len | num_audio_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 16 | 320 | 2053 | 40 | 0.4159 | 0.0954 |
| 2 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 32 | 320 | 449 | 40 | 0.4463 | 0.0970 |
| 3 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 2 | 960 | 512 | 120 | 0.3540 | 0.0181 |
| 4 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 2 | 1920 | 1249 | 240 | 0.3394 | 0.0357 |
| 5 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 1 | 3200 | 1721 | 400 | 0.3379 | 0.0299 |
| 6 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 1 | 640 | 256 | 80 | 0.3432 | 0.0079 |
| 7 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 2 | 2240 | 1801 | 280 | 0.3339 | 0.0418 |
| 8 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 16 | 960 | 919 | 120 | 0.4092 | 0.1418 |
| 9 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 32 | 1280 | 8192 | 160 | 1.2079 | 0.7249 |
| 10 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 8 | 1920 | 1973 | 240 | 0.4170 | 0.1418 |
| 11 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 4 | 1280 | 1024 | 160 | 0.3463 | 0.0478 |
| 12 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 16 | 640 | 373 | 80 | 0.3691 | 0.0953 |
| 13 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 8 | 1280 | 853 | 160 | 0.3725 | 0.0953 |
| 14 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 2 | 320 | 131 | 40 | 0.3534 | 0.0079 |
| 15 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 64 | 640 | 541 | 80 | 0.6104 | 0.3798 |
| 16 | 80 | 512 | 1024 | 1024 | 15 | 40 | 4096 | 16 | 64 | 64 | 320 | 1087 | 40 | 0.5797 | 0.2350 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    audio_seq_len = axes_and_scalars["audio_seq_len"]
    text_seq_len = axes_and_scalars["text_seq_len"]
    num_audio_tokens = axes_and_scalars["num_audio_tokens"]
    
    encoder_input_dim = 80
    encoder_hidden_dim = 512
    encoder_output_dim = 1024
    projector_hidden_size = 1024
    num_queries = 40
    text_hidden_size = 4096
    
    # Audio features
    input_features = torch.randn(batch_size, audio_seq_len, encoder_input_dim, dtype=torch.bfloat16, device=device)
    
    # LM embeddings
    lm_embeddings = torch.randn(batch_size, text_seq_len, text_hidden_size, dtype=torch.bfloat16, device=device)
    
    # Audio token positions - generate valid positions within text_seq_len
    # Ensure positions are sorted and unique per batch
    audio_token_positions = torch.zeros(batch_size, num_audio_tokens, dtype=torch.int64, device=device)
    for b in range(batch_size):
        positions = torch.randperm(text_seq_len, device=device)[:num_audio_tokens].sort().values
        audio_token_positions[b] = positions
    
    # Encoder weights
    encoder_input_weight = torch.randn(encoder_hidden_dim, encoder_input_dim, dtype=torch.bfloat16, device=device) * 0.02
    encoder_input_bias = torch.zeros(encoder_hidden_dim, dtype=torch.bfloat16, device=device)
    encoder_out_weight = torch.randn(encoder_output_dim, encoder_hidden_dim, dtype=torch.bfloat16, device=device) * 0.02
    encoder_out_bias = torch.zeros(encoder_output_dim, dtype=torch.bfloat16, device=device)
    encoder_out_mid_weight = torch.randn(encoder_hidden_dim, encoder_output_dim, dtype=torch.bfloat16, device=device) * 0.02
    encoder_out_mid_bias = torch.zeros(encoder_hidden_dim, dtype=torch.bfloat16, device=device)
    
    # Q-Former weights
    learnable_queries = torch.randn(1, num_queries, projector_hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    qformer_q_proj_weight = torch.randn(projector_hidden_size, projector_hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    qformer_q_proj_bias = torch.zeros(projector_hidden_size, dtype=torch.bfloat16, device=device)
    qformer_k_proj_weight = torch.randn(projector_hidden_size, encoder_hidden_dim, dtype=torch.bfloat16, device=device) * 0.02
    qformer_k_proj_bias = torch.zeros(projector_hidden_size, dtype=torch.bfloat16, device=device)
    qformer_v_proj_weight = torch.randn(projector_hidden_size, encoder_hidden_dim, dtype=torch.bfloat16, device=device) * 0.02
    qformer_v_proj_bias = torch.zeros(projector_hidden_size, dtype=torch.bfloat16, device=device)
    qformer_out_proj_weight = torch.randn(projector_hidden_size, projector_hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    qformer_out_proj_bias = torch.zeros(projector_hidden_size, dtype=torch.bfloat16, device=device)
    
    # Projector weights
    projector_weight = torch.randn(text_hidden_size, projector_hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    projector_bias = torch.zeros(text_hidden_size, dtype=torch.bfloat16, device=device)
    
    return {
        "input_features": input_features,
        "lm_embeddings": lm_embeddings,
        "audio_token_positions": audio_token_positions,
        "encoder_input_weight": encoder_input_weight,
        "encoder_input_bias": encoder_input_bias,
        "encoder_out_weight": encoder_out_weight,
        "encoder_out_bias": encoder_out_bias,
        "encoder_out_mid_weight": encoder_out_mid_weight,
        "encoder_out_mid_bias": encoder_out_mid_bias,
        "learnable_queries": learnable_queries,
        "qformer_q_proj_weight": qformer_q_proj_weight,
        "qformer_q_proj_bias": qformer_q_proj_bias,
        "qformer_k_proj_weight": qformer_k_proj_weight,
        "qformer_k_proj_bias": qformer_k_proj_bias,
        "qformer_v_proj_weight": qformer_v_proj_weight,
        "qformer_v_proj_bias": qformer_v_proj_bias,
        "qformer_out_proj_weight": qformer_out_proj_weight,
        "qformer_out_proj_bias": qformer_out_proj_bias,
        "projector_weight": projector_weight,
        "projector_bias": projector_bias,
    }

@torch.no_grad()
def run(
    input_features: torch.Tensor,
    lm_embeddings: torch.Tensor,
    audio_token_positions: torch.Tensor,
    encoder_input_weight: torch.Tensor,
    encoder_input_bias: torch.Tensor,
    encoder_out_weight: torch.Tensor,
    encoder_out_bias: torch.Tensor,
    encoder_out_mid_weight: torch.Tensor,
    encoder_out_mid_bias: torch.Tensor,
    learnable_queries: torch.Tensor,
    qformer_q_proj_weight: torch.Tensor,
    qformer_q_proj_bias: torch.Tensor,
    qformer_k_proj_weight: torch.Tensor,
    qformer_k_proj_bias: torch.Tensor,
    qformer_v_proj_weight: torch.Tensor,
    qformer_v_proj_bias: torch.Tensor,
    qformer_out_proj_weight: torch.Tensor,
    qformer_out_proj_bias: torch.Tensor,
    projector_weight: torch.Tensor,
    projector_bias: torch.Tensor,
):
    batch_size = input_features.shape[0]
    text_seq_len = lm_embeddings.shape[1]
    num_audio_tokens = audio_token_positions.shape[1]
    
    encoder_hidden_dim = 512
    projector_hidden_size = 1024
    text_hidden_size = 4096
    window_size = 15
    num_queries = 40
    qformer_num_heads = 16
    head_dim = projector_hidden_size // qformer_num_heads
    
    # Stage 1: Encoder input projection
    hidden_states = F.linear(
        input_features.to(torch.float32),
        encoder_input_weight.to(torch.float32),
        encoder_input_bias.to(torch.float32)
    )
    
    # Simulate mid-layer CTC feedback (simplified encoder processing)
    hidden_states_mid = F.linear(
        hidden_states,
        encoder_out_weight.to(torch.float32),
        encoder_out_bias.to(torch.float32)
    )
    softmax_out = F.softmax(hidden_states_mid, dim=-1)
    feedback = F.linear(
        softmax_out,
        encoder_out_mid_weight.to(torch.float32),
        encoder_out_mid_bias.to(torch.float32)
    )
    hidden_states = hidden_states + feedback
    
    encoder_output = hidden_states  # [batch_size, audio_seq_len, 512]
    
    # Stage 2: Windowed Q-Former downsampling
    seq_len = encoder_output.shape[1]
    nblocks = math.ceil(seq_len / window_size)
    pad = nblocks * window_size - seq_len
    
    # Pad to window boundaries
    if pad > 0:
        encoder_output_padded = F.pad(encoder_output, (0, 0, 0, pad), "constant", 0)
    else:
        encoder_output_padded = encoder_output
    
    # Reshape into windows
    encoder_output_windowed = encoder_output_padded.view(
        batch_size * nblocks, window_size, encoder_hidden_dim
    )
    
    # Broadcast learnable queries for all windows
    queries = learnable_queries.expand(batch_size * nblocks, -1, -1).to(torch.float32)
    
    # Q-Former cross-attention
    Q = F.linear(
        queries,
        qformer_q_proj_weight.to(torch.float32),
        qformer_q_proj_bias.to(torch.float32)
    )
    K = F.linear(
        encoder_output_windowed,
        qformer_k_proj_weight.to(torch.float32),
        qformer_k_proj_bias.to(torch.float32)
    )
    V = F.linear(
        encoder_output_windowed,
        qformer_v_proj_weight.to(torch.float32),
        qformer_v_proj_bias.to(torch.float32)
    )
    
    # Reshape for multi-head attention
    bsz_nblocks = batch_size * nblocks
    Q = Q.view(bsz_nblocks, num_queries, qformer_num_heads, head_dim).transpose(1, 2)
    K = K.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)
    V = V.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bsz_nblocks, num_queries, projector_hidden_size
    )
    
    # Output projection
    query_output = F.linear(
        attn_output,
        qformer_out_proj_weight.to(torch.float32),
        qformer_out_proj_bias.to(torch.float32)
    )
    
    # Reshape back to batch dimension
    query_output = query_output.view(
        batch_size, nblocks * num_queries, projector_hidden_size
    )
    
    # Project to LM dimension
    audio_embeddings = F.linear(
        query_output,
        projector_weight.to(torch.float32),
        projector_bias.to(torch.float32)
    )  # [batch_size, nblocks * num_queries, text_hidden_size]
    
    # Stage 3: Scatter audio embeddings into LM embeddings
    # Clone lm_embeddings to avoid modifying input
    fused_embeddings = lm_embeddings.to(torch.float32).clone()
    
    # Scatter audio embeddings at specified positions
    # audio_token_positions: [batch_size, num_audio_tokens]
    # We need to scatter audio_embeddings[:, :num_audio_tokens, :] to these positions
    for b in range(batch_size):
        positions = audio_token_positions[b]  # [num_audio_tokens]
        # Take only the first num_audio_tokens from audio_embeddings
        audio_emb = audio_embeddings[b, :num_audio_tokens, :]  # [num_audio_tokens, text_hidden_size]
        fused_embeddings[b].index_copy_(0, positions, audio_emb)
    
    return fused_embeddings.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.068985 ms |
| - | Scoring Baseline | 0.500000 | 0.423215 ms |
| - | Reference Implementation | 0.065747 | 7.614227 ms |
