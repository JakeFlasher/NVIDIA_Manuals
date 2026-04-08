## Description

Vision patch merger that performs spatial shuffling to merge patches (spatial_merge_size=2), applies layer normalization, and processes through a 2-layer MLP with GELU activation. Converts fine-grained vision tokens into coarser representations by merging 2x2 spatial patches.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden | [num_patches, hidden_size] | bfloat16 |
| grid_thw | [num_grids, three] | int64 |
| ln_weight | [hidden_size] | bfloat16 |
| ln_bias | [hidden_size] | bfloat16 |
| fc1_weight | [hidden_size_expanded, hidden_size_expanded] | bfloat16 |
| fc1_bias | [hidden_size_expanded] | bfloat16 |
| fc2_weight | [out_hidden_size, hidden_size_expanded] | bfloat16 |
| fc2_bias | [out_hidden_size] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_merged_patches, out_hidden_size] | bfloat16 |

| # | hidden_size | spatial_merge_size | out_hidden_size | hidden_size_expanded | three | num_patches | num_merged_patches | num_grids | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 2 | 3584 | 6144 | 3 | 4096 | 1024 | 4 | 0.2076 | 0.0680 |
| 2 | 1536 | 2 | 3584 | 6144 | 3 | 1024 | 256 | 1 | 0.1091 | 0.0173 |
| 3 | 1536 | 2 | 3584 | 6144 | 3 | 576 | 144 | 1 | 0.1036 | 0.0099 |
| 4 | 1536 | 2 | 3584 | 6144 | 3 | 3072 | 768 | 3 | 0.1680 | 0.0511 |
| 5 | 1536 | 2 | 3584 | 6144 | 3 | 1600 | 400 | 1 | 0.1254 | 0.0268 |
| 6 | 1536 | 2 | 3584 | 6144 | 3 | 8192 | 2048 | 8 | 0.3337 | 0.1356 |
| 7 | 1536 | 2 | 3584 | 6144 | 3 | 6400 | 1600 | 1 | 0.2649 | 0.1060 |
| 8 | 1536 | 2 | 3584 | 6144 | 3 | 32768 | 8192 | 8 | 0.9716 | 0.5410 |
| 9 | 1536 | 2 | 3584 | 6144 | 3 | 12288 | 3072 | 3 | 0.4513 | 0.2031 |
| 10 | 1536 | 2 | 3584 | 6144 | 3 | 7168 | 1792 | 7 | 0.3035 | 0.1187 |
| 11 | 1536 | 2 | 3584 | 6144 | 3 | 16384 | 4096 | 4 | 0.5678 | 0.2707 |
| 12 | 1536 | 2 | 3584 | 6144 | 3 | 65536 | 16384 | 4 | 1.8445 | 1.0817 |
| 13 | 1536 | 2 | 3584 | 6144 | 3 | 1024 | 256 | 4 | 0.1112 | 0.0173 |
| 14 | 1536 | 2 | 3584 | 6144 | 3 | 2048 | 512 | 2 | 0.1428 | 0.0342 |
| 15 | 1536 | 2 | 3584 | 6144 | 3 | 512 | 128 | 2 | 0.0996 | 0.0088 |

```python
import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs with valid grid_thw that matches num_patches."""
    num_patches = axes_and_scalars["num_patches"]
    num_merged_patches = axes_and_scalars["num_merged_patches"]
    num_grids = axes_and_scalars["num_grids"]
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584
    merge_size = 2
    eps = 1e-6
    
    # Generate grid_thw such that total patches matches num_patches
    # Each grid contributes T * H * W patches, and H, W must be divisible by merge_size
    patches_per_grid = num_patches // num_grids
    
    # Find valid H, W that are divisible by merge_size
    # We want T * H * W = patches_per_grid
    # Let's use T=1 for simplicity and find H, W
    sqrt_patches = int(math.sqrt(patches_per_grid))
    # Round to nearest multiple of merge_size
    h = (sqrt_patches // merge_size) * merge_size
    if h == 0:
        h = merge_size
    w = (patches_per_grid // h // merge_size) * merge_size
    if w == 0:
        w = merge_size
    t = patches_per_grid // (h * w)
    if t == 0:
        t = 1
    
    # Adjust to match exactly
    actual_patches_per_grid = t * h * w
    
    # Create grid_thw tensor
    grid_thw = torch.zeros((num_grids, 3), dtype=torch.int64, device=device)
    remaining_patches = num_patches
    for i in range(num_grids):
        if i == num_grids - 1:
            # Last grid takes remaining patches
            patches_for_this = remaining_patches
        else:
            patches_for_this = actual_patches_per_grid
        
        # Find valid T, H, W for this grid
        sqrt_p = int(math.sqrt(patches_for_this))
        h_i = (sqrt_p // merge_size) * merge_size
        if h_i == 0:
            h_i = merge_size
        w_i = (patches_for_this // h_i // merge_size) * merge_size
        if w_i == 0:
            w_i = merge_size
        t_i = patches_for_this // (h_i * w_i)
        if t_i == 0:
            t_i = 1
        
        grid_thw[i, 0] = t_i
        grid_thw[i, 1] = h_i
        grid_thw[i, 2] = w_i
        remaining_patches -= t_i * h_i * w_i
    
    hidden = torch.randn(num_patches, hidden_size, dtype=torch.bfloat16, device=device)
    ln_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    ln_bias = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(hidden_size_expanded, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc1_bias = torch.randn(hidden_size_expanded, dtype=torch.bfloat16, device=device)
    fc2_weight = torch.randn(out_hidden_size, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc2_bias = torch.randn(out_hidden_size, dtype=torch.bfloat16, device=device)
    
    return {
        "hidden": hidden,
        "grid_thw": grid_thw,
        "ln_weight": ln_weight,
        "ln_bias": ln_bias,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "eps": eps,
    }

@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    """
    Vision patch merger with spatial shuffling and MLP.
    
    1. Apply layer normalization (pre-shuffle)
    2. Spatial shuffle to merge 2x2 patches
    3. Two-layer MLP with GELU activation
    """
    merge_size = 2
    hidden_size = 1536
    hidden_size_expanded = 6144
    
    # Step 1: Layer normalization (pre-shuffle, on hidden_size dimension)
    hidden_fp32 = hidden.to(torch.float32)
    mean = hidden_fp32.mean(dim=-1, keepdim=True)
    var = hidden_fp32.var(dim=-1, keepdim=True, unbiased=False)
    hidden_norm = (hidden_fp32 - mean) / torch.sqrt(var + eps)
    hidden_norm = hidden_norm * ln_weight.to(torch.float32) + ln_bias.to(torch.float32)
    hidden_norm = hidden_norm.to(torch.bfloat16)
    
    # Step 2: Spatial shuffle to merge patches
    shuffled_patches = []
    offset = 0
    
    for i in range(grid_thw.shape[0]):
        t = grid_thw[i, 0].item()
        h = grid_thw[i, 1].item()
        w = grid_thw[i, 2].item()
        
        num_patches_this = t * h * w
        patches = hidden_norm[offset:offset + num_patches_this]
        
        h_merged = h // merge_size
        w_merged = w // merge_size
        
        # Reshape to (T, H/merge_size, merge_size, W/merge_size, merge_size, C)
        patches = patches.view(t, h_merged, merge_size, w_merged, merge_size, hidden_size)
        
        # Permute to (T, H/merge_size, W/merge_size, merge_size, merge_size, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5)
        
        # Flatten spatial merge groups: (T * H/merge_size * W/merge_size, merge_size^2 * C)
        patches = patches.reshape(t * h_merged * w_merged, hidden_size_expanded)
        
        shuffled_patches.append(patches)
        offset += num_patches_this
    
    hidden_shuffled = torch.cat(shuffled_patches, dim=0)
    
    # Step 3: Two-layer MLP with GELU
    # FC1: (num_merged_patches, hidden_size_expanded) @ (hidden_size_expanded, hidden_size_expanded).T
    hidden_fc1 = torch.nn.functional.linear(hidden_shuffled, fc1_weight, fc1_bias)
    
    # GELU activation
    hidden_gelu = torch.nn.functional.gelu(hidden_fc1)
    
    # FC2: (num_merged_patches, hidden_size_expanded) @ (out_hidden_size, hidden_size_expanded).T
    output = torch.nn.functional.linear(hidden_gelu, fc2_weight, fc2_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.069811 ms |
| - | Scoring Baseline | 0.500000 | 0.250539 ms |
| - | Reference Implementation | 0.230563 | 0.645714 ms |
