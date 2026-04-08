## Description

Complex 3D position ID calculation for multi-modal inputs (text, images, videos) that computes temporal, height, and width position indices for rotary embeddings. Handles variable-length sequences with different grid sizes, performs spatial merging calculations, and creates position deltas for rope indexing.
| Name | Shape | Dtype |
| --- | --- | --- |
| input_ids | [batch_size, seq_len] | int64 |
| image_grid_thw | [total_num_images, 3] | int64 |
| video_grid_thw | [total_num_videos, 3] | int64 |
| second_per_grid_ts | [total_num_videos] | float32 |
| attention_mask | [batch_size, seq_len] | int64 |

| Name | Shape | Dtype |
| --- | --- | --- |
| position_ids | [3, batch_size, seq_len] | int64 |
| mrope_position_deltas | [batch_size, 1] | int64 |

| # | spatial_merge_size | image_token_id | video_token_id | vision_start_token_id | batch_size | seq_len | total_num_images | total_num_videos | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 151655 | 151656 | 151652 | 2 | 2048 | 2 | 2 | 0.1465 | 0.0004 |
| 2 | 2 | 151655 | 151656 | 151652 | 4 | 512 | 2 | 2 | 0.1437 | 0.0004 |
| 3 | 2 | 151655 | 151656 | 151652 | 4 | 1024 | 4 | 0 | 0.1328 | 0.0004 |
| 4 | 2 | 151655 | 151656 | 151652 | 2 | 512 | 0 | 2 | 0.1171 | 0.0004 |
| 5 | 2 | 151655 | 151656 | 151652 | 8 | 512 | 0 | 4 | 0.1574 | 0.0004 |
| 6 | 2 | 151655 | 151656 | 151652 | 16 | 373 | 0 | 4 | 0.1885 | 0.0004 |
| 7 | 2 | 151655 | 151656 | 151652 | 64 | 256 | 0 | 8 | 0.2441 | 0.0004 |
| 8 | 2 | 151655 | 151656 | 151652 | 32 | 256 | 8 | 0 | 0.1645 | 0.0004 |
| 9 | 2 | 151655 | 151656 | 151652 | 1 | 2048 | 1 | 1 | 0.1236 | 0.0004 |
| 10 | 2 | 151655 | 151656 | 151652 | 32 | 512 | 0 | 8 | 0.2973 | 0.0004 |
| 11 | 2 | 151655 | 151656 | 151652 | 1 | 256 | 1 | 0 | 0.0839 | 0.0004 |
| 12 | 2 | 151655 | 151656 | 151652 | 8 | 256 | 4 | 2 | 0.1370 | 0.0004 |
| 13 | 2 | 151655 | 151656 | 151652 | 4 | 256 | 2 | 0 | 0.1278 | 0.0004 |
| 14 | 2 | 151655 | 151656 | 151652 | 32 | 449 | 8 | 0 | 0.2101 | 0.0004 |
| 15 | 2 | 151655 | 151656 | 151652 | 1 | 4096 | 3 | 2 | 0.1505 | 0.0004 |
| 16 | 2 | 151655 | 151656 | 151652 | 8 | 691 | 0 | 4 | 0.1624 | 0.0004 |

```python
import torch
from typing import Tuple

def get_inputs(
    axes_and_scalars: dict,
    device: torch.device
) -> dict:
    """Generate custom inputs for multimodal rope position calculation.
    
    The key insight is that total_num_images and total_num_videos are the TOTAL
    counts across ALL batches, not per-batch counts. We need to distribute them
    across batches and ensure the sequence length is sufficient.
    """
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    total_num_images = axes_and_scalars["total_num_images"]
    total_num_videos = axes_and_scalars["total_num_videos"]
    
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    
    # Generate input_ids with random tokens
    input_ids = torch.randint(0, 50000, (batch_size, seq_len), dtype=torch.int64, device=device)
    
    # Generate grid dimensions for images (t=1 for images)
    if total_num_images > 0:
        image_t = torch.ones(total_num_images, dtype=torch.int64, device=device)
        image_h = torch.randint(1, 3, (total_num_images,), dtype=torch.int64, device=device) * spatial_merge_size
        image_w = torch.randint(1, 3, (total_num_images,), dtype=torch.int64, device=device) * spatial_merge_size
        image_grid_thw = torch.stack([image_t, image_h, image_w], dim=1)
    else:
        image_grid_thw = torch.zeros((0, 3), dtype=torch.int64, device=device)
    
    # Generate grid dimensions for videos (t >= 1)
    if total_num_videos > 0:
        video_t = torch.randint(1, 3, (total_num_videos,), dtype=torch.int64, device=device)
        video_h = torch.randint(1, 3, (total_num_videos,), dtype=torch.int64, device=device) * spatial_merge_size
        video_w = torch.randint(1, 3, (total_num_videos,), dtype=torch.int64, device=device) * spatial_merge_size
        video_grid_thw = torch.stack([video_t, video_h, video_w], dim=1)
        second_per_grid_ts = torch.rand(total_num_videos, dtype=torch.float32, device=device) * 0.5 + 0.5
    else:
        video_grid_thw = torch.zeros((0, 3), dtype=torch.int64, device=device)
        second_per_grid_ts = torch.zeros(0, dtype=torch.float32, device=device)
    
    # Distribute vision tokens across batches
    # Simple distribution: round-robin assignment
    images_per_batch = [0] * batch_size
    videos_per_batch = [0] * batch_size
    
    for i in range(total_num_images):
        images_per_batch[i % batch_size] += 1
    for i in range(total_num_videos):
        videos_per_batch[i % batch_size] += 1
    
    # Place vision tokens in input_ids for each batch
    global_img_idx = 0
    global_vid_idx = 0
    
    for b in range(batch_size):
        pos = 5  # Start after some text tokens
        num_imgs_this_batch = images_per_batch[b]
        num_vids_this_batch = videos_per_batch[b]
        
        # Place images for this batch
        for _ in range(num_imgs_this_batch):
            if global_img_idx >= total_num_images:
                break
            t, h, w = image_grid_thw[global_img_idx]
            llm_h = h.item() // spatial_merge_size
            llm_w = w.item() // spatial_merge_size
            num_tokens = t.item() * llm_h * llm_w
            
            if pos + num_tokens + 2 < seq_len:
                input_ids[b, pos] = vision_start_token_id
                input_ids[b, pos + 1] = image_token_id
                # Fill vision token positions with placeholder
                for j in range(num_tokens):
                    if pos + 2 + j < seq_len:
                        input_ids[b, pos + 2 + j] = 0
                pos += 2 + num_tokens + 3  # Add gap
            global_img_idx += 1
        
        # Place videos for this batch
        for _ in range(num_vids_this_batch):
            if global_vid_idx >= total_num_videos:
                break
            t, h, w = video_grid_thw[global_vid_idx]
            llm_h = h.item() // spatial_merge_size
            llm_w = w.item() // spatial_merge_size
            num_tokens = t.item() * llm_h * llm_w
            
            if pos + num_tokens + 2 < seq_len:
                input_ids[b, pos] = vision_start_token_id
                input_ids[b, pos + 1] = video_token_id
                for j in range(num_tokens):
                    if pos + 2 + j < seq_len:
                        input_ids[b, pos + 2 + j] = 0
                pos += 2 + num_tokens + 3
            global_vid_idx += 1
    
    # Generate attention mask (all ones)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    
    return {
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,
        "attention_mask": attention_mask,
    }

@torch.no_grad()
def run(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw: torch.Tensor,
    second_per_grid_ts: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D position IDs for multi-modal rotary embeddings."""
    
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    tokens_per_second = 2.0
    
    batch_size, seq_length = input_ids.shape
    device = input_ids.device
    dtype = input_ids.dtype
    
    # Initialize position IDs
    position_ids = torch.ones(
        3, batch_size, seq_length,
        dtype=dtype,
        device=device,
    )
    
    mrope_position_deltas = []
    
    total_num_images = image_grid_thw.shape[0]
    total_num_videos = video_grid_thw.shape[0]
    
    if total_num_images > 0 or total_num_videos > 0:
        attention_mask_bool = attention_mask == 1
        
        # Global indices for images and videos across all batches
        image_index, video_index = 0, 0
        
        for i in range(batch_size):
            input_id_seq = input_ids[i]
            # Apply attention mask
            input_id_seq = input_id_seq[attention_mask_bool[i]]
            
            # Find vision token locations in this batch
            vision_start_indices = torch.argwhere(
                input_id_seq == vision_start_token_id
            ).squeeze(1)
            
            if len(vision_start_indices) == 0:
                # Pure text sequence
                valid_length = attention_mask_bool[i].sum()
                position_ids[:, i, :valid_length] = torch.arange(
                    valid_length, dtype=dtype, device=device
                ).unsqueeze(0).expand(3, -1)
                mrope_position_deltas.append(0)
                continue
            
            vision_tokens = input_id_seq[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            
            input_tokens = input_id_seq.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums.item(), video_nums.item()
            
            # Process each vision token in this batch
            for _ in range(image_nums + video_nums):
                # Find next image or video token
                ed_image = len(input_tokens) + 1
                ed_video = len(input_tokens) + 1
                
                if remain_images > 0:
                    try:
                        ed_image = input_tokens.index(image_token_id, st)
                    except ValueError:
                        ed_image = len(input_tokens) + 1
                
                if remain_videos > 0:
                    try:
                        ed_video = input_tokens.index(video_token_id, st)
                    except ValueError:
                        ed_video = len(input_tokens) + 1
                
                # Determine if image or video
                if ed_image < ed_video:
                    if image_index < total_num_images:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0.0
                        image_index += 1
                    else:
                        break
                    remain_images -= 1
                    ed = ed_image
                else:
                    if video_index < total_num_videos:
                        t, h, w = video_grid_thw[video_index]
                        second_per_grid_t = second_per_grid_ts[video_index].item()
                        video_index += 1
                    else:
                        break
                    remain_videos -= 1
                    ed = ed_video
                
                # Calculate grid dimensions after spatial merge
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                
                # Add text positions before vision token
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device, dtype=dtype)
                    .view(1, -1).expand(3, -1) + st_idx
                )
                
                # Create 3D position indices for vision tokens
                # Temporal positions
                range_tensor = torch.arange(llm_grid_t, device=device, dtype=torch.float32).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                t_index = time_tensor.long().flatten()
                
                # Height positions
                h_index = torch.arange(llm_grid_h, device=device, dtype=dtype)
                h_index = h_index.view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                
                # Width positions
                w_index = torch.arange(llm_grid_w, device=device, dtype=dtype)
                w_index = w_index.view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                
                llm_pos_ids_list.append(
                    torch.stack([t_index.to(dtype), h_index, w_index]) + text_len + st_idx
                )
                
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            
            # Add remaining text positions
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device, dtype=dtype)
                    .view(1, -1).expand(3, -1) + st_idx
                )
            
            # Concatenate all positions
            llm_positions = torch.cat(llm_pos_ids_list, dim=1)
            
            # Assign to position_ids
            valid_len = min(llm_positions.shape[1], attention_mask_bool[i].sum().item())
            position_ids[:, i, :valid_len] = llm_positions[:, :valid_len]
            
            # Calculate position delta
            mrope_position_deltas.append(
                llm_positions.max().item() + 1 - len(input_id_seq)
            )
        
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=device, dtype=dtype
        ).unsqueeze(1)
    else:
        # No vision tokens - standard 1D positions
        position_ids_1d = attention_mask.long().cumsum(-1) - 1
        position_ids_1d.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - seq_length
    
    return position_ids, mrope_position_deltas
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000400 ms |
| 1st place | AKO4ALL_L2 | 0.846672 | 0.028134 ms |
| 2nd place | Nestor_Qin | 0.657647 | 0.082432 ms |
| - | Scoring Baseline | 0.500000 | 0.154750 ms |
| - | Reference Implementation | 0.078081 | 2.122780 ms |
