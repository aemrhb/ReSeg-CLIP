# ReSeg-CLIP 🛰️
## Zero-shot Semantic Segmentation for Remote Sensing with CLIP

This repo contains the official implementation of ReSeg-CLIP, a training-free framework for Open-Vocabulary Semantic Segmentation (OVSS) in aerial and satellite imagery.

## 🔥 Key Features

- **Hierarchical Attention Masking**: Combines initial layers, neighbor-based attention, and SAM-generated attention masks
- **Model Composition**: Integrates CLIP vision encoder with Segment Anything Model (SAM) for enhanced segmentation
- **No Training Required**: Zero-shot inference using pre-trained CLIP and SAM models

## 📋 Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- See `requirements.txt` for full dependency list

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/aemrhb/ReSeg-CLIP.git
cd ReSeg-CLIP
```

2. Create a virtual environment (recommended):
```bash
conda create -n reseg-clip python=3.10 -y
conda activate reseg-clip
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: Make sure to install PyTorch with CUDA support from [pytorch.org](https://pytorch.org) if you plan to use GPU acceleration.

## 📁 Data Structure

Your input data should be organized as follows:

```
DATA_ROOT/
  ├── tile_001/
  │   ├── R_G_B_IR.png    # Input image (RGB or RGB-IR)
  │   └── L.png            # Ground-truth label mask (optional, for evaluation)
  ├── tile_002/
  │   ├── R_G_B_IR.png
  │   └── L.png
  └── ...
```

- `R_G_B_IR.png`: The input image to be segmented (filename can be customized in config)
- `L.png`: Ground-truth segmentation mask (required for computing metrics)

## ⚙️ Configuration

Edit `config.yml` to configure the model and data paths:

### Key Configuration Sections:

1. **Text Queries**: Define the text prompts that CLIP will use to identify different classes
```yaml
text_queries:
  - "Aerial photo of a street in the city"
  - "Aerial photo of a building in the city"
  # ... more queries
```

2. **Class Mapping**: Map original class indices to semantic categories
```yaml
class_mapping:
  "0": "Artificial Surface"
  "5": "Building"
  # ... more mappings
```

3. **Input/Output Paths**:
```yaml
input:
  dir: "/path/to/your/data"
  target_image_name: "R_G_B_IR.png"

output:
  base_dir: "/path/to/output"
  visualizations_subdir: "visualizations"
```

4. **Model Configuration**:
```yaml
model:
  clip_model_name: "ViT-L/14"  # CLIP model variant
  clip_checkpoint: "/path/to/clip_checkpoint.pt"
  sam_checkpoint: "/path/to/sam_vit_h_4b8939.pth"
  img_size: 224
  patch_size: 14
  num_classes: 6
```

5. **Attention Settings**: Control hierarchical attention masking
```yaml
attention:
  num_initial_layers: 18              # Layers without attention restrictions
  num_neighbor_attention_layers: 2    # Layers with neighbor-based attention
  num_sam_attention_layers: 4         # Layers with SAM-based attention
  sam_param_sets:                     # SAM mask generation parameters
    - points_per_side: 4
      pred_iou_thresh: 0.5
      # ... more parameters
```

## 🏃 Running Inference

Run the inference script with your configuration file:

```bash
python infernce.py --config config.yml
```

### What the script does:

1. **Loads Models**: 
   - CLIP vision encoder (from checkpoint)
   - SAM (Segment Anything Model)

2. **Processes Images**:
   - Iterates through all subdirectories in the input directory
   - Finds images matching `target_image_name`
   - Processes images in patches with sliding window approach
   - Applies hierarchical attention masks to CLIP layers
   - Generates pixel-wise semantic predictions

3. **Computes Metrics** (if ground-truth available):
   - Overall accuracy
   - Per-class F1 scores
   - Per-class IoU (Intersection over Union)
   - Mean F1 and Mean IoU

4. **Saves Outputs**:
   - `pred_mask.png`: Color-coded prediction mask
   - `pred_overlay.png`: Predictions overlaid on original image
   - `gt_mask.png`: Ground-truth color mask (if available)
   - `gt_overlay.png`: Ground-truth overlaid on original image (if available)

All outputs are saved in `output.base_dir/visualizations_subdir/`.

## 📊 Output Metrics

The script prints cumulative metrics after processing each image:
- **Accuracy**: Overall pixel classification accuracy
- **Mean F1-Score (Mf1)**: Average F1-score across all classes
- **Mean IoU (Miou)**: Average Intersection over Union across all classes
- **Per-class metrics**: Individual F1 and IoU for each semantic class

## 🔧 How It Works

1. **Patch Processing**: Large images are divided into overlapping patches for efficient processing
2. **Attention Masking**: 
   - Initial layers process without restrictions
   - SAM generates segmentation masks with varying parameters
   - Neighbor-based attention guides local feature aggregation
3. **CLIP Encoding**: Text queries and image patches are encoded using CLIP
4. **Similarity Matching**: Pixel-wise cosine similarity between visual and text features determines class assignments
5. **Aggregation**: Overlapping patch predictions are averaged to produce final segmentation

## 📝 Notes

- The code automatically uses GPU if available, otherwise falls back to CPU
- Memory management: CUDA cache is cleared between images to handle large datasets
- The script processes all matching images recursively in the input directory

## 📄 Citation

If you use this code, please cite the paper (coming soon).

## 📜 License

[Add your license information here]
