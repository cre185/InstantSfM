import numpy as np
import os
import time
import tqdm
import concurrent.futures

from instantsfm.utils import database

# Local pair ID encoding for matcher indices (not related to ViewGraph pair keys)
C_MAX_INT = 2**31 - 1

def PairId2Ids(pair_id):
    return (pair_id % C_MAX_INT, pair_id // C_MAX_INT)

def Ids2PairId(id1, id2):
    return (id1 * C_MAX_INT + id2 if id1 < id2 else id2 * C_MAX_INT + id1)

def GenerateDatabase(image_path, database_path, feature_handler_name, config, single_camera=False, camera_per_folder=False, sequential_num=None):
    # colmap support from command line. ensure colmap is installed
    import subprocess
    # Force COLMAP to run SIFT on CPU to avoid GPU/OpenGL context requirements inside containers
    colmap_env = os.environ.copy()
    colmap_env["CUDA_VISIBLE_DEVICES"] = ""
    feature_extractor_cmd = [
        'colmap', 'feature_extractor',
        '--image_path', image_path,
        '--database_path', database_path,
        '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        '--ImageReader.single_camera', '1' if single_camera else '0',
        '--ImageReader.single_camera_per_folder', '1' if (not single_camera and camera_per_folder) else '0',
        '--SiftExtraction.use_gpu', '0'
    ]
    exhaustive_matcher_cmd = [
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path,
        '--SiftMatching.use_gpu', '0'
    ]
    sequential_matcher_cmd = [
        'colmap', 'sequential_matcher',
        '--database_path', database_path,
        '--SiftMatching.use_gpu', '0'
    ]
    use_exhaustive = True
    matcher_cmd = exhaustive_matcher_cmd if use_exhaustive else sequential_matcher_cmd

    try:
        print(f"Feature extraction started for {image_path} by {feature_extractor_cmd}")
        subprocess.run(feature_extractor_cmd, check=True, env=colmap_env)
        print(f"Feature extraction completed for {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction: {e}")
    try:
        subprocess.run(matcher_cmd, check=True, env=colmap_env)
        print(f"Exhaustive matching completed for {database_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during exhaustive matching: {e}")
    return