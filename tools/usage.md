# Tools Usage Guide

This document describes utility tools provided with InstantSfM for preprocessing and auxiliary tasks.

## Video Depth Anything

The `video_depth_anything.py` script generates metric depth maps from image sequences using the [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) model. InstantSfM currently supports only metric depth, so make sure to use the metric depth models. Note that Video Depth Anything requires the input images to be a continuous image sequence (e.g., frames extracted from a video).  

### Setup

You can follow the [official instructions](https://github.com/DepthAnything/Video-Depth-Anything) to set up Video Depth Anything, or follow the steps below:  

**1. Clone Video Depth Anything**

Clone the Video Depth Anything repository into the `external/` directory:
```bash
cd external
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
cd Video-Depth-Anything
```

**2. Install Dependencies**

Install the required Python packages:
```bash
conda create -n vda python=3.10
conda activate vda
pip install -r requirements.txt
```
Then install pytorch and xformers as per your CUDA version. For example, for CUDA 12.1:
```bash
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
```


**3. Download Model Checkpoints**

Create a `checkpoints/` directory and download the pretrained weights:
```bash
mkdir -p checkpoints
cd checkpoints
```

For **metric depth** (recommended):
```bash
# Large model (best quality)
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth

# Base model (balanced)
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth

# Small model (fastest)
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth
```

### Usage

**Basic Command**

Process a dataset directory containing images:
```bash
python tools/video_depth_anything.py \
    --data_path /path/to/dataset \
    --encoder vitl
```

The script will:
1. Search for image folders recursively in `data_path`
2. Process each folder containing images
3. Save depth maps to `data_path/depth_vda/` by default

**Directory Structure**

The input directory should have exactly the same structure as required by InstantSfM. Use the same `data_path` as for InstantSfM's processing. Output depth maps will be saved in a subdirectory named `depth_vda/`.
