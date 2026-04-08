## Description

Computes 3D rotary position embeddings for multimodal inputs by generating position indices from grid_thw (temporal, height, width) parameters, performing bilinear interpolation for variable resolution support, and creating interleaved MRoPE embeddings. Combines grid-based position index generation, learned position embedding lookup with bilinear interpolation weights, spatial permutation to merge_size blocks, and rotary embedding frequency computation with interleaving.
| Name | Shape | Dtype |
| --- | --- | --- |
| grid_thw | [num_images, 3] | int64 |
| pos_embed_weight | [num_position_embeddings, hidden_size] | float32 |
| inv_freq | [quarter_head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| patch_pos_embeds | [total_tokens, hidden_size] | float32 |
| cos_embeddings | [total_tokens, head_dim] | float32 |
| sin_embeddings | [total_tokens, head_dim] | float32 |

| # | spatial_merge_size | num_position_embeddings | num_grid_per_side | hidden_size | num_heads | head_dim | quarter_head_dim | num_images | total_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 4 | 64 | 0.2130 | 0.0005 |
| 2 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 3 | 768 | 0.2091 | 0.0014 |
| 3 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 5 | 320 | 0.2079 | 0.0008 |
| 4 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 10 | 640 | 0.2098 | 0.0010 |
| 5 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 4 | 256 | 0.2101 | 0.0007 |
| 6 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 7 | 448 | 0.2070 | 0.0009 |
| 7 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 4 | 1024 | 0.2063 | 0.0016 |
| 8 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 2 | 2048 | 0.2072 | 0.0037 |
| 9 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 16 | 4096 | 0.2309 | 0.0043 |
| 10 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 1 | 1024 | 0.2272 | 0.0025 |
| 11 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 6 | 96 | 0.2148 | 0.0006 |
| 12 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 2 | 512 | 0.2166 | 0.0012 |
| 13 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 2 | 128 | 0.2163 | 0.0006 |
| 14 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 1 | 256 | 0.2075 | 0.0009 |
| 15 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 16 | 1024 | 0.2138 | 0.0014 |
| 16 | 2 | 1225 | 35 | 1536 | 12 | 128 | 32 | 7 | 112 | 0.2218 | 0.0006 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_images = axes_and_scalars["num_images"]
    total_tokens = axes_and_scalars["total_tokens"]
    num_position_embeddings = axes_and_scalars["num_position_embeddings"]
    hidden_size = axes_and_scalars["hidden_size"]
    head_dim = axes_and_scalars["head_dim"]
    spatial_merge_size = axes_and_scalars["spatial_merge_size"]
    rope_theta = 10000.0
    
    # Generate grid_thw values that sum to total_tokens
    # Each entry is [t, h, w] where h and w are divisible by spatial_merge_size
    # total_tokens = sum(t * h * w) for all entries
    grid_thw = torch.zeros((num_images, 3), dtype=torch.int64, device=device)
    
    remaining_tokens = total_tokens
    for i in range(num_images):
        if i == num_images - 1:
            # Last image gets remaining tokens
            # Find t, h, w such that t * h * w = remaining_tokens
            # Use simple factorization
            t = 1
            # Try to find h, w divisible by 2
            hw = remaining_tokens
            # Find factors
            h = 2
            while hw % h != 0 and h < hw:
                h += 2
            if hw % h == 0:
                w = hw // h
                if w % 2 != 0:
                    # Adjust
                    h = 4
                    while hw % h != 0 and h < hw:
                        h += 2
                    if hw % h == 0:
                        w = hw // h
            else:
                h = 2
                w = 2
                t = remaining_tokens // 4
                if t == 0:
                    t = 1
            grid_thw[i] = torch.tensor([t, h, w], dtype=torch.int64)
        else:
            # Generate random valid dimensions
            t = max(1, min(4, remaining_tokens // (num_images - i) // 16))
            h = max(2, min(34, (remaining_tokens // (num_images - i) // t) // 4)) // 2 * 2
            w = max(2, min(34, (remaining_tokens // (num_images - i) // t // h))) // 2 * 2
            if h < 2:
                h = 2
            if w < 2:
                w = 2
            tokens_this = t * h * w
            if tokens_this > remaining_tokens:
                t = 1
                h = 2
                w = 2
                tokens_this = 4
            grid_thw[i] = torch.tensor([t, h, w], dtype=torch.int64)
            remaining_tokens -= tokens_this
    
    # Position embedding weights
    pos_embed_weight = torch.randn(num_position_embeddings, hidden_size, dtype=torch.float32, device=device)
    
    # Inverse frequencies for rotary embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32, device=device) / head_dim))
    
    return {
        "grid_thw": grid_thw,
        "pos_embed_weight": pos_embed_weight,
        "inv_freq": inv_freq
    }

@torch.no_grad()
def run(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor, inv_freq: torch.Tensor):
    device = grid_thw.device
    dtype = pos_embed_weight.dtype
    
    # Constants
    spatial_merge_size = 2
    num_grid_per_side = 35
    hidden_size = pos_embed_weight.shape[1]
    head_dim = inv_freq.shape[0] * 4
    mrope_section = [24, 20, 20]
    
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    
    # Part 1: Bilinear Interpolation Position Embeddings
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h.item(), device=device)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w.item(), device=device)
        
        h_idxs_floor = h_idxs.long()
        w_idxs_floor = w_idxs.long()
        h_idxs_ceil = (h_idxs_floor + 1).clamp(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs_floor + 1).clamp(max=num_grid_per_side - 1)
        
        dh = h_idxs - h_idxs_floor.float()
        dw = w_idxs - w_idxs_floor.float()
        
        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side
        
        indices = [
            (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
            (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
        ]
        
        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]
        
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())
    
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=dtype, device=device)
    
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    
    patch_pos_embeds = patch_pos_embeds.split([h.item() * w.item() for h, w in zip(grid_hs, grid_ws)])
    
    patch_pos_embeds_permute = []
    merge_size = spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t.item(), 1)
        pos_embed = (
            pos_embed.view(t.item(), h.item() // merge_size, merge_size,
                          w.item() // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    
    # Part 2: Rotary Position Embeddings with MRoPE
    max_hw = int(grid_thw[:, 1:].max().item())
    
    seq = torch.arange(max_hw, device=device, dtype=torch.float32)
    freqs = torch.outer(seq, inv_freq)
    
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
    
    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height.item() // merge_size, width.item() // merge_size
        
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)
        
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        
        coords = torch.stack((row_idx, col_idx), dim=-1)
        
        if num_frames > 1:
            coords = coords.repeat(num_frames.item(), 1)
        
        num_tokens = coords.shape[0]
        pos_ids[offset:offset + num_tokens] = coords
        offset += num_tokens
    
    embeddings = freqs[pos_ids]
    embeddings = embeddings.flatten(1)
    
    freqs_3d = embeddings.unsqueeze(0).expand(3, -1, -1).clone()
    
    freqs_t = freqs_3d[0]
    for dim, offset_dim in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset_dim, length, 3)
        freqs_t[..., idx] = freqs_3d[dim, ..., idx]
    
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    cos_embeddings = emb.cos()
    sin_embeddings = emb.sin()
    
    return patch_pos_embeds, cos_embeddings, sin_embeddings
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001133 ms |
| - | Scoring Baseline | 0.500000 | 0.213589 ms |
| - | Reference Implementation | 0.075360 | 3.044405 ms |
