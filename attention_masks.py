"""
Attention mask generation functions for CLIP with SAM-based segmentation.
"""
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import torch
import random


def set_attention_masks(model, masks):
    """
    Set attention masks for all transformer blocks in the CLIP vision model.
    
    Args:
        model: CLIP model with visual.transformer.resblocks
        masks: List of attention masks, one per transformer block
    """
    blocks = model.visual.transformer.resblocks
    if len(masks) != len(blocks):
        raise ValueError(f"Expected {len(blocks)} masks, got {len(masks)}")
    for block, m in zip(blocks, masks):
        block.attn_mask = m


def create_combined_attention_masks(
    img_size, 
    patch_size, 
    num_initial_layers, 
    num_neighbor_attention_layers, 
    num_sam_attention_layers, 
    sam_model, 
    image, 
    sam_param_sets,  # List of parameter sets for SAM masks
    max_random_attn=0.0
):
    """
    Create a list of attention masks. The first n masks are based on neighbor expansion,
    and the remaining num_masks - n masks are based on SAM masks generated with varying parameters.
    
    Args:
        img_size: Size of the input image (assumed square)
        patch_size: Size of each patch
        num_initial_layers: Number of initial layers without attention restrictions
        num_neighbor_attention_layers: Number of layers with neighbor-based attention
        num_sam_attention_layers: Number of layers with SAM-based attention (should match len(sam_param_sets))
        sam_model: SAM model instance
        image: Input image array
        sam_param_sets: List of parameter dictionaries for SAM mask generation
        max_random_attn: Fraction of random tokens to add as attention (optional)
    
    Returns:
        List of attention masks (torch.Tensor) for each transformer layer
    """
    # Helper function to generate no-attention masks
    def generate_no_attention_masks():
        num_patches_per_dim = img_size // patch_size
        num_patches = num_patches_per_dim ** 2
        total_tokens = num_patches + 1  # Including the class token
        masks = [torch.full((total_tokens, total_tokens), False, dtype=torch.bool) for _ in range(num_initial_layers)]
        return masks

    # Helper function to generate neighbor-expansion masks
    def generate_neighbor_masks():
        num_patches_per_dim = img_size // patch_size
        num_patches = num_patches_per_dim ** 2
        total_tokens = num_patches + 1  # Including the class token
        directions = [
            (0, 0)
        ]
        max_levels = len(directions)
        levels_per_mask = max_levels / num_neighbor_attention_layers
        masks = []

        for mask_idx in range(num_neighbor_attention_layers):
            level = int(levels_per_mask * (mask_idx + 1))
            level = min(level, max_levels)

            mask = torch.full((total_tokens, total_tokens), True, dtype=torch.bool)
            for i in range(num_patches_per_dim):
                for j in range(num_patches_per_dim):
                    patch_idx = i * num_patches_per_dim + j
                    mask[patch_idx + 1, patch_idx + 1] = False
                    for dx, dy in directions[:level]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < num_patches_per_dim and 0 <= nj < num_patches_per_dim:
                            neighbor_idx = ni * num_patches_per_dim + nj
                            mask[patch_idx + 1, neighbor_idx + 1] = False

            num_random_patches = int(max_random_attn * num_patches)
            random_patches = random.sample(range(1, total_tokens), num_random_patches)
            mask[0, 0] = False
            for patch in random_patches:
                mask[0, patch] = False
                mask[patch, 0] = False

            masks.append(mask)
        return masks

    # Generate SAM-based masks with varying parameters
    def generate_sam_masks():
        masks = []
        for params in sam_param_sets:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=params.get('points_per_side', 16),
                pred_iou_thresh=params.get('pred_iou_thresh', 0.4),
                stability_score_thresh=params.get('stability_score_thresh', 0.4),
                crop_n_layers=params.get('crop_n_layers', 1),
                crop_n_points_downscale_factor=params.get('crop_n_points_downscale_factor', 2),
                min_mask_region_area=params.get('min_mask_region_area', 100),
            )
            with torch.no_grad():
                sam_masks = mask_generator.generate(image)

            segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            for i, mask in enumerate(sam_masks):
                segmentation_map[mask['segmentation']] = i + 1

            h, w = image.shape[:2]
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            patch_segmentation_ids = np.zeros((num_patches_h, num_patches_w), dtype=np.int32)

            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = segmentation_map[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                    unique_ids, counts = np.unique(patch, return_counts=True)
                    patch_segmentation_ids[i, j] = unique_ids[np.argmax(counts)]

            num_patches = num_patches_h * num_patches_w
            total_tokens = num_patches + 1
            
            # Flatten the patch IDs (e.g., shape 256)
            ids_flat = patch_segmentation_ids.flatten()

            # Generate connectivity matrix: True where IDs match
            # This is a (256, 256) matrix
            connectivity = (ids_flat[:, None] == ids_flat[None, :])

            # Convert to the CLIP mask format (total tokens = num_patches + 1)
            mask = torch.ones((total_tokens, total_tokens), dtype=torch.bool)

            # Fill the patch-to-patch attention (1: starts after CLS token)
            # In CLIP masks, False means "allowed to attend"
            mask[1:, 1:] = ~torch.from_numpy(connectivity)

            # Ensure CLS token attends to itself
            mask[0, 0] = False
            masks.append(mask)
        return masks

    # Combine all types of masks
    no_attention_masks = generate_no_attention_masks()
    neighbor_masks = generate_neighbor_masks()
    sam_masks = generate_sam_masks()
    return no_attention_masks + sam_masks + neighbor_masks

