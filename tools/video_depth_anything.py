#!/usr/bin/env python3
"""
Video Depth Anything Processing Script

This script processes image sequences to generate metric depth maps using Video Depth Anything.
It's designed to work with InstantSFM datasets but can be used independently.

Usage:
    python tools/video_depth_anything.py \\
        --data_path /path/to/dataset \\
        --vda_path external/Video-Depth-Anything \\
        --encoder vitl
"""

import os
import sys
import glob
from pathlib import Path
from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
import cv2


def setup_vda_path(vda_path):
    """Add Video Depth Anything to Python path and return utilities.
    
    Args:
        vda_path: Path to Video-Depth-Anything directory
        
    Returns:
        Tuple of (VideoDepthAnything class, read_video_frames function)
    """
    vda_path = Path(vda_path).resolve()
    if not vda_path.exists():
        raise FileNotFoundError(f"Video-Depth-Anything not found at: {vda_path}")
    
    # Remove current directory from sys.path to avoid conflicts with this script's name
    script_dir = str(Path(__file__).parent.resolve())
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    
    # Add VDA directory to path (not parent) to import video_depth_anything package
    vda_path_str = str(vda_path)
    if vda_path_str not in sys.path:
        sys.path.insert(0, vda_path_str)
    
    # Now we can import
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
        from utils.dc_utils import read_video_frames
        return VideoDepthAnything, read_video_frames
    except ImportError as e:
        print(f"Error importing Video Depth Anything modules: {e}")
        print(f"Make sure Video-Depth-Anything is properly set up at: {vda_path}")
        print(f"\nChecking directory structure:")
        print(f"  VDA path: {vda_path}")
        print(f"  video_depth_anything module: {vda_path / 'video_depth_anything'}")
        print(f"  utils module: {vda_path / 'utils'}")
        raise


def create_video_from_images(image_folder, output_video_path, fps=30):
    """Create a video from a folder of images.
    
    Args:
        image_folder: Path to folder containing images
        output_video_path: Path to save the output video
        fps: Frames per second for the video
    
    Returns:
        Tuple of (success: bool, image_filenames: list of str)
        image_filenames contains basenames of images in processing order
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return False, []
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Extract basenames for later use
    image_basenames = [os.path.basename(f) for f in image_files]
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Failed to read first image: {image_files[0]}")
        return False, []
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write all images to video
    print(f"Creating video from {len(image_files)} images...")
    for img_path in tqdm(image_files, desc=f"Processing {Path(image_folder).name}"):
        img = cv2.imread(img_path)
        if img is not None:
            # Resize if necessary
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            out.write(img)
    
    out.release()
    print(f"✓ Video created: {output_video_path}")
    return True, image_basenames


def save_depth_maps(depths, output_folder, image_filenames=None, save_npz=True, save_npy=True):
    """Save depth maps in various formats.
    
    Args:
        depths: List or array of depth maps [N, H, W]
        output_folder: Folder to save depth maps
        image_filenames: Optional list of original image filenames (basenames)
        save_npz: Save all depths in a single compressed npz file
        save_npy: Save individual depth maps as npy files
    
    Returns:
        Path to saved files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Save compressed npz for all depths
    if save_npz:
        npz_path = os.path.join(output_folder, 'depths.npz')
        np.savez_compressed(npz_path, depths=depths)
        print(f"✓ Saved compressed depths to {npz_path}")
    
    # Save individual depth maps as npy files
    if save_npy:
        npy_folder = os.path.join(output_folder, 'npy')
        os.makedirs(npy_folder, exist_ok=True)
        
        print(f"Saving {len(depths)} individual depth maps...")
        for i, depth in enumerate(tqdm(depths, desc="Saving npy files")):
            # Use original image filename if available, otherwise use index
            if image_filenames and i < len(image_filenames):
                # Replace image extension with .npy
                base_name = os.path.splitext(image_filenames[i])[0]
                npy_path = os.path.join(npy_folder, f'{base_name}.npy')
            else:
                npy_path = os.path.join(npy_folder, f'depth_{i:06d}.npy')
            np.save(npy_path, depth)
        
        print(f"✓ Saved {len(depths)} individual depth maps to {npy_folder}")
    
    return output_folder


def save_visualizations(depths, output_folder, num_samples=5):
    """Save depth visualization samples.
    
    Args:
        depths: Array of depth maps
        output_folder: Folder to save visualizations
        num_samples: Number of visualization samples to save
    """
    vis_folder = os.path.join(output_folder, 'visualizations')
    os.makedirs(vis_folder, exist_ok=True)
    
    # Save a few visualization samples
    num_vis_samples = min(num_samples, len(depths))
    vis_indices = np.linspace(0, len(depths)-1, num_vis_samples, dtype=int)
    
    print(f"Saving {num_vis_samples} visualization samples...")
    for idx in vis_indices:
        depth = depths[idx]
        
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        vis_path = os.path.join(vis_folder, f'depth_vis_{idx:06d}.png')
        cv2.imwrite(vis_path, depth_colored)
    
    print(f"✓ Saved visualization samples to {vis_folder}")


def process_folder_with_video_depth(model, read_video_frames_func, image_folder, output_folder, 
                                     encoder_type='vitl', input_size=518, max_res=1280, 
                                     fps=30, device='cuda', fp32=False):
    """Process a folder of images with Video Depth Anything.
    
    Args:
        model: VideoDepthAnything model instance
        read_video_frames_func: Function to read video frames
        image_folder: Path to folder containing images
        output_folder: Path to save depth results
        encoder_type: Encoder type (vits, vitb, vitl)
        input_size: Input size for the model
        max_res: Maximum resolution
        fps: Frames per second for video
        device: Device to run inference on
        fp32: Use float32 precision instead of float16
    
    Returns:
        depths array if successful, None otherwise
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create temporary video file
    temp_video_path = os.path.join(output_folder, 'temp_video.mp4')
    success, image_filenames = create_video_from_images(image_folder, temp_video_path, fps)
    if not success:
        return None
    
    # Process video with Video Depth Anything
    print(f"Processing video with Video Depth Anything ({encoder_type})...")
    try:
        # Read video frames - using correct parameter names: process_length, target_fps, max_res
        frames, target_fps = read_video_frames_func(temp_video_path, process_length=-1, target_fps=-1, max_res=max_res)
        
        # Infer depth
        depths, output_fps = model.infer_video_depth(frames, target_fps, input_size=input_size, device=device, fp32=fp32)
        
        print(f"✓ Processed {len(depths)} frames with Video Depth Anything")
        
        # Save depth maps with original filenames
        save_depth_maps(depths, output_folder, image_filenames=image_filenames, save_npz=True, save_npy=True)
        
        # Save visualizations
        save_visualizations(depths, output_folder, num_samples=5)
        
        return depths
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up temporary video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Cleaned up temporary video")


def find_image_folders(data_path):
    """Find all folders containing images.
    
    Args:
        data_path: Root path to search
        
    Returns:
        List of Path objects for folders containing images
    """
    image_folders = []
    data_path = Path(data_path)
    
    # Check if data_path itself contains images
    has_images = any(
        f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        for f in data_path.iterdir() if f.is_file()
    )
    if has_images:
        image_folders.append(data_path)
        return image_folders
    
    # Otherwise search subdirectories
    for root, dirs, files in os.walk(data_path):
        # Check if this folder contains images
        has_images = any(
            f.lower().endswith(('.jpg', '.jpeg', '.png'))
            for f in files
        )
        if has_images:
            image_folders.append(Path(root))
    
    return image_folders


def main():
    """Main function to run Video Depth Anything on image sequences."""
    parser = ArgumentParser(description='Run Video Depth Anything on image sequences')
    parser.add_argument('--data_path', required=True, help='Path to the data folder containing image sequences')
    parser.add_argument('--output_path', help='Path to save depth results (default: data_path/depth_vda)')
    parser.add_argument('--vda_path', default='external/Video-Depth-Anything',
                        help='Path to Video-Depth-Anything directory (default: external/Video-Depth-Anything)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type (default: vitl)')
    parser.add_argument('--input_size', type=int, default=518, help='Input size for the model (default: 518)')
    parser.add_argument('--max_res', type=int, default=1280, help='Maximum resolution (default: 1280)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for video (default: 30)')
    parser.add_argument('--fp32', action='store_true', help='Use float32 precision instead of float16')
    parser.add_argument('--relative', action='store_true', help='Use relative depth model instead of metric depth (default: metric depth)')
    parser.add_argument('--checkpoint_path', help='Path to Video Depth Anything checkpoint file (default: auto-detect in vda_path/checkpoints)')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return 1
    
    # Setup Video Depth Anything
    print("Setting up Video Depth Anything...")
    try:
        VideoDepthAnything, read_video_frames = setup_vda_path(args.vda_path)
    except Exception as e:
        print(f"Failed to setup Video Depth Anything: {e}")
        return 1
    
    # Setup output path
    if args.output_path:
        output_base = Path(args.output_path)
    else:
        output_base = data_path / 'depth_vda'
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Determine if using metric or relative depth
    use_metric = not args.relative
    depth_type = 'metric' if use_metric else 'relative'
    print(f"Using {depth_type} depth model")
    
    # Setup checkpoint path
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        vda_path = Path(args.vda_path).resolve()
        checkpoint_prefix = 'metric_video_depth_anything' if use_metric else 'video_depth_anything'
        checkpoint_name = f'{checkpoint_prefix}_{args.encoder}.pth'
        checkpoint_path = vda_path / 'checkpoints' / checkpoint_name
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print(f"\nPlease download the checkpoint first:")
        if use_metric:
            print(f"cd {Path(args.vda_path).resolve() / 'checkpoints'}")
            print(f"wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_{args.encoder}.pth")
        else:
            print(f"cd {Path(args.vda_path).resolve()}")
            print(f"bash get_weights.sh")
        return 1
    
    # Load model
    print(f"Loading Video Depth Anything model ({args.encoder}, {depth_type})...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = VideoDepthAnything(**model_configs[args.encoder], metric=use_metric)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
    model = model.to(device).eval()
    print("✓ Model loaded successfully")
    
    # Find all image folders
    print(f"\nScanning for image folders in {data_path}...")
    image_folders = find_image_folders(data_path)
    
    if not image_folders:
        print(f"No image folders found in {data_path}")
        return 1
    
    print(f"Found {len(image_folders)} folder(s) to process")
    
    # Process each folder
    success_count = 0
    
    for folder in image_folders:
        # Determine output folder
        if folder == data_path:
            # Root folder contains images directly
            output_folder = output_base
        else:
            # Preserve relative structure
            folder_name = folder.relative_to(data_path)
            output_folder = output_base / folder_name
        
        print(f"\n{'='*60}")
        print(f"Processing: {folder.relative_to(data_path) if folder != data_path else folder.name}")
        print(f"Output: {output_folder}")
        print(f"{'='*60}")
        
        depths = process_folder_with_video_depth(
            model,
            read_video_frames,
            str(folder),
            str(output_folder),
            encoder_type=args.encoder,
            input_size=args.input_size,
            max_res=args.max_res,
            fps=args.fps,
            device=device,
            fp32=args.fp32
        )
        
        if depths is not None:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(image_folders)} folder(s) processed successfully")
    print(f"Results saved to: {output_base}")
    print(f"{'='*60}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    return 0 if success_count == len(image_folders) else 1


if __name__ == '__main__':
    sys.exit(main())
