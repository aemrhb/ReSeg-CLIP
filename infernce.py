from segment_anything import SamPredictor, sam_model_registry
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import math
import os
import sys
import argparse
import yaml

# Add the parent directory of the clip package to the Python path so we can import the local CLIP package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip

# Import attention mask functions from separate module
from attention_masks import set_attention_masks, create_combined_attention_masks


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="CLIP + SAM inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- config sections ---
    text_queries = cfg["text_queries"]
    # YAML keys are strings, convert to int for class_mapping
    class_mapping = {int(k): v for k, v in cfg["class_mapping"].items()}
    category_to_index = cfg["category_to_index"]

    input_cfg = cfg["input"]
    input_dir = input_cfg["dir"]
    target_image_name = input_cfg.get("target_image_name", "R_G_B_IR.png")

    model_cfg = cfg["model"]
    clip_model_name = model_cfg.get("clip_model_name", "ViT-L/14")
    clip_type_path = model_cfg["clip_checkpoint"]
    sam_checkpoint = model_cfg["sam_checkpoint"]
    img_size = model_cfg["img_size"]
    patch_size = model_cfg["patch_size"]
    num_classes = model_cfg["num_classes"]

    attention_cfg = cfg["attention"]
    num_initial_layers = attention_cfg["num_initial_layers"]
    num_neighbor_attention_layers = attention_cfg["num_neighbor_attention_layers"]
    num_sam_attention_layers = attention_cfg["num_sam_attention_layers"]
    max_random_attn = attention_cfg.get("max_random_attn", 0.0)
    sam_param_sets = attention_cfg["sam_param_sets"]

    output_cfg = cfg["output"]
    output_base_dir = output_cfg["base_dir"]
    visualizations_subdir = output_cfg.get("visualizations_subdir", "visualizations")

    # --- devices and models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    # Load CLIP
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    ckpt = torch.load(clip_type_path, map_location="cpu")
    clip_model.load_state_dict(ckpt, strict=False)
    clip_model.eval()
                    
    all_gt_masks = []
    all_predicted_classes = []
    # Placeholder for dataset-level metrics
    all_gt_masks = []
    all_predicted_classes = []
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="none")
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="none")

    # Iterate over all images in the directory
    entries = sorted(os.listdir(input_dir))
    for root, dirs, files in tqdm(os.walk(input_dir), desc="Processing directories"):
        for filename in tqdm(files, desc="Processing files in directory", leave=False):
            torch.cuda.empty_cache()
            if filename == target_image_name:
                file_path = os.path.join(root, filename)
                print(f"Found image file: {file_path}")
                file_path = os.path.join(root, filename)
                print(f"Found image file: {file_path}")
                path = os.path.join(root, filename)
                print("path", path)
                # Define target size
                target_size = img_size

                # Process image
                image = Image.open(path)
                image = np.array(image)
                image = image[:, :, :3]
                # image = image[0:1000, 0:1000]
                image = Image.fromarray(image)
                image_width, image_height = image.size

                # Compute the number of patches
                num_patches_x = math.ceil(image_width / target_size)
                num_patches_y = math.ceil(image_height / target_size)

                # Iterate through patches
                # Initialize accumulators for probabilities and overlap counts
                length_of_texts = len(text_queries)
                prob_accumulator = np.zeros((image_height, image_width, length_of_texts), dtype=np.float32)
                overlap_count = np.zeros((image_height, image_width), dtype=np.int32)

                # Update the stride to 1/4 of the target size
                stride_x = target_size
                stride_y = target_size

                # Iterate through patches with overlaps
                for i in tqdm(
                    range(0, image_height - target_size + stride_y, stride_y),
                    desc="Processing patches (rows)",
                    leave=False,
                ):
                    for j in range(0, image_width - target_size + stride_x, stride_x):
                        print("run", image_height - target_size + stride_y)
                        left = j
                        upper = i
                        right = min(left + target_size, image_width)
                        lower = min(upper + target_size, image_height)

                        # Crop the patch
                        patch = image.crop((left, upper, right, lower))
                        # Add padding if the patch is smaller than target size
                        pad_width = target_size - (right - left)
                        pad_height = target_size - (lower - upper)
                        padding = (0, 0, pad_width, pad_height)
                        patch = ImageOps.expand(patch, padding)
                        patch_sam = np.array(patch)

                        attention_masks = create_combined_attention_masks(
                            img_size=img_size,
                            patch_size=patch_size,
                            num_initial_layers=num_initial_layers,
                            num_neighbor_attention_layers=num_neighbor_attention_layers,
                            num_sam_attention_layers=num_sam_attention_layers,
                            sam_model=sam,
                            image=patch_sam,
                            sam_param_sets=sam_param_sets,
                            max_random_attn=max_random_attn,
                        )

                        set_attention_masks(clip_model, attention_masks)

                        # Process text queries
                        text = clip.tokenize(text_queries).to(device)
                        with torch.no_grad():
                            text_feature = clip_model.encode_text(text)
                        text_feature /= text_feature.norm(dim=-1, keepdim=True)

                        patch = preprocess(patch).unsqueeze(0).to(device)

                        # Encode image
                        with torch.no_grad():
                            visual_features, attn_weights = clip_model.encode_image(patch)
                        cls_token = visual_features[0, 0:1]
                        cls_token /= cls_token.norm(dim=-1, keepdim=True)

                        # Cosine similarity
                        cosine_similarities = (cls_token @ text_feature.T).detach().cpu().numpy()[0]
                        text_probs = (100.0 * cls_token @ text_feature.T).softmax(dim=-1).detach().cpu().numpy()[0]

                        # Visual features reshaping and interpolation
                        visual_features1 = visual_features[0, 1:].detach().cpu()

                        patch_grid_size = int(visual_features1.shape[0] ** 0.5)  # Assuming a square grid of patches
                        H, W = img_size, img_size
                        visual_features_reshaped = (
                            visual_features1.reshape(patch_grid_size, patch_grid_size, -1)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )  # Shape: [1, feature_dim, patch_grid_size, patch_grid_size]
                        visual_features_interp = F.interpolate(
                            visual_features_reshaped, size=(H, W), mode="bilinear", align_corners=False
                        )  # Shape: [1, feature_dim, H, W]
                        text_feature = text_feature.unsqueeze(0)  # Shape: [1, num_texts, dim]
                        visual_features_interp = F.normalize(visual_features_interp, p=2, dim=1)  # Normalize along feature_dim
                        text_feature = F.normalize(text_feature, p=2, dim=-1).detach().cpu()

                        # Pixel-wise cosine similarity
                        visual_features_interp_flat = (
                            visual_features_interp.squeeze(0).permute(1, 2, 0).reshape(-1, visual_features_interp.shape[1])
                        )
                        cosine_sim = F.cosine_similarity(
                            visual_features_interp_flat.unsqueeze(1), text_feature, dim=-1
                        )
                        class_probabilities = F.softmax(cosine_sim, dim=-1).detach().cpu().numpy()

                        # Reshape to the original image size
                        predicted_probs = class_probabilities.reshape(img_size, img_size, -1)

                        crop_left = padding[0]
                        crop_upper = padding[1]
                        crop_right = patch_sam.shape[1] - padding[2]  # width - right padding
                        crop_lower = patch_sam.shape[0] - padding[3]  # height - bottom padding

                        # Crop the image to undo the padding
                        predicted_probs = predicted_probs[crop_upper:crop_lower, crop_left:crop_right]

                        # Accumulate probabilities
                        prob_accumulator[upper:lower, left:right, :] += predicted_probs
                        overlap_count[upper:lower, left:right] += 1

                # Average probabilities in overlapping areas
                final_probs = prob_accumulator / np.maximum(overlap_count[..., None], 1)  # Avoid division by zero
                final_predictions = np.argmax(final_probs, axis=-1)  # Get final class predictions

                predicted_classes = np.vectorize(lambda x: category_to_index[class_mapping[x]])(final_predictions)

                # Load GT mask
                gt_path = os.path.join(os.path.dirname(path), "L.png")
                print("gt_path", gt_path)
                gt_mask = np.array(Image.open(gt_path))
                # gt_mask = gt_mask[0:1000, 0:1000]
                gt_mask = torch.tensor(gt_mask, dtype=torch.long).cpu().numpy()

                # After your patch prediction:
                print("predicted_classes", predicted_classes.shape)
                print("gt_mask", gt_mask.shape)
                # Update cumulative metrics
                accuracy_metric.update(
                    torch.from_numpy(predicted_classes).flatten(), torch.from_numpy(gt_mask).flatten()
                )
                f1_metric.update(torch.from_numpy(predicted_classes).flatten(), torch.from_numpy(gt_mask).flatten())
                miou_metric.update(
                    torch.from_numpy(predicted_classes).flatten(), torch.from_numpy(gt_mask).flatten()
                )

                # Cumulative metrics
                cumulative_accuracy = accuracy_metric.compute().item()
                cumulative_f1_per_class = f1_metric.compute()  # Shape: (num_classes,)
                cumulative_miou_per_class = miou_metric.compute()  # Shape: (num_classes,)

                # Mean values
                cumulative_mf1 = cumulative_f1_per_class.mean().item()
                cumulative_miou = cumulative_miou_per_class.mean().item()

                # Print cumulative metrics
                print("model", clip_type_path)
                print(f"--- Cumulative Metrics after {filename} ---")
                print(f"Cumulative Accuracy: {cumulative_accuracy:.4f}")
                print(f"Cumulative Mean F1-Score (Mf1): {cumulative_mf1:.4f}")
                print(f"Cumulative Mean IoU (Miou): {cumulative_miou:.4f}")

                # Print class-wise F1 and IoU
                for class_idx in range(num_classes):
                    print(
                        f"Class {class_idx}: Cumulative F1 = {cumulative_f1_per_class[class_idx]:.4f}, "
                        f"Cumulative IoU = {cumulative_miou_per_class[class_idx]:.4f}"
                    )
                print("============================================\n")

                # --- after you have `predicted_classes` (H×W ints 0..num_classes-1) and original image as PIL image ---

                # 1) convert your PIL image to a numpy array
                orig_np = np.array(image).astype(np.uint8)  # shape (H, W, 3)

                # 2) define one color per class (you can customize these)
                palette = np.array(
                    [
                        [107, 142, 35],   # class 0
                        [102, 102, 156],  # class 1
                        [128, 64, 128],   # class 2
                        [0, 0, 142],      # class 3
                        [0, 0, 0],        # class 4
                        [0, 0, 256],      # class 5
                    ],
                    dtype=np.uint8,
                )

                # 3) build the color mask for predictions
                pred_color_mask = palette[predicted_classes]  # shape (H, W, 3)
                gt_color_mask = palette[gt_mask]  # same for ground truth

                # 4) blend masks with original image
                alpha = 0.5  # transparency of mask
                pred_overlay = (orig_np * (1 - alpha) + pred_color_mask * alpha).astype(np.uint8)
                gt_overlay = (orig_np * (1 - alpha) + gt_color_mask * alpha).astype(np.uint8)

                # 5) save out all four:
                base = output_base_dir
                out_dir = os.path.join(base, visualizations_subdir)
                os.makedirs(out_dir, exist_ok=True)

                Image.fromarray(pred_color_mask).save(os.path.join(out_dir, "pred_mask.png"))
                Image.fromarray(pred_overlay).save(os.path.join(out_dir, "pred_overlay.png"))
                Image.fromarray(gt_color_mask).save(os.path.join(out_dir, "gt_mask.png"))
                Image.fromarray(gt_overlay).save(os.path.join(out_dir, "gt_overlay.png"))


if __name__ == "__main__":
    main()


