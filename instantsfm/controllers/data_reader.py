import numpy as np
import time
import os
import cv2
import glob

from instantsfm.scene.defs import Images, ImagePair, Cameras, ConfigurationType, CameraModelId, ViewGraph
from instantsfm.utils.database import COLMAPDatabase, blob_to_array
from instantsfm.utils.depth_sample import sample_depth_at_pixel

class PathInfo:
    def __init__(self):
        self.image_path = ""
        self.database_path = ""
        self.output_path = ""
        self.database_exists = False
        self.depth_path = ""
        self.fixed_relative_poses_path = ""
        self.semantics_path = ""
        self.record_path = ""

def ReadData(path) -> PathInfo:
    path_info = PathInfo()
    if os.path.exists(os.path.join(path, 'images')):
        # COLMAP format
        path_info.image_path = os.path.join(path, 'images')
    elif os.path.exists(os.path.join(path, 'color')):  
        # ScanNet format
        path_info.image_path = os.path.join(path, 'color')
    else:
        # used in camera_per_folder mode
        path_info.image_path = path
    
    path_info.database_path = os.path.join(path, 'database.db')
    path_info.output_path = os.path.join(path, 'sparse')
    path_info.database_exists = os.path.exists(path_info.database_path)
    
    # Check for depth directory (supports both 'depth' and 'depth_vda')
    if os.path.exists(os.path.join(path, 'depth')):
        path_info.depth_path = os.path.join(path, 'depth')
    elif os.path.exists(os.path.join(path, 'depth_vda')):
        path_info.depth_path = os.path.join(path, 'depth_vda')
    
    if os.path.exists(os.path.join(path, 'semantics')):
        path_info.semantics_path = os.path.join(path, 'semantics')
    path_info.record_path = os.path.join(path, 'record')

    # Check for gt relative poses for multi-camera rig
    rel_poses_path = os.path.join(path, 'camera_relative_poses.txt')
    if os.path.exists(rel_poses_path):
        path_info.fixed_relative_poses_path = rel_poses_path

    # Check for semantics directory
    if os.path.exists(os.path.join(path, 'semantics')):
        path_info.semantics_path = os.path.join(path, 'semantics')

    return path_info

def ReadColmapDatabase(path):
    start_time = time.time()
    view_graph = ViewGraph()
    db = COLMAPDatabase.connect(path)
    
    # Read images into temporary dict for initial processing
    # Create temporary image data structures
    images_dict = {}
    for id, filename, cam_id in db.execute("SELECT image_id, name, camera_id FROM images"):
        images_dict[id] = {
            'id': id,
            'filename': filename,
            'cam_id': cam_id,
            'features': np.array([]),
            'is_registered': False,
            'cluster_id': -1,
            'world2cam': np.eye(4),
            'depths': np.array([]),
            'semantics': None,  # Will be loaded if available
            'features_undist': np.array([]),
            'point3d_ids': [],
            'num_points3d': 0
        }
    # group images by their folder names
    image_folders = {}
    for image_data in images_dict.values():
        folder_name = os.path.dirname(image_data['filename'])
        if folder_name not in image_folders:
            image_folders[folder_name] = []
        image_folders[folder_name].append(image_data)

    # Create temporary camera data structures
    camera_records = {}
    for id, model_id, width, height, params, prior_focal_length in db.execute("SELECT * FROM cameras"):
        camera_records[id] = {
            'id': id,
            'model_id': CameraModelId(model_id),
            'width': width,
            'height': height,
            'params': blob_to_array(params, np.float64),
            'has_prior_focal_length': prior_focal_length > 0
        }
    
    keypoints = [(image_id, blob_to_array(data, np.float32, (-1, cols)))
                 for image_id, cols, data in db.execute("SELECT image_id, cols, data FROM keypoints") if not data is None]
    for image_id, data in keypoints:
        images_dict[image_id]['features'] = data[:, :2]

    query = """
    SELECT m.pair_id, m.data, t.data, t.config, t.F, t.E, t.H
    FROM matches AS m
    INNER JOIN two_view_geometries AS t ON m.pair_id = t.pair_id
    """
    matches_and_geometries = db.execute(query)
    image_pairs = {}
    invalid_count = 0

    for group in matches_and_geometries:
        pair_id, raw_data_blob, verified_data_blob, config, F_blob, E_blob, H_blob = group
        raw_data = None if raw_data_blob is None else blob_to_array(raw_data_blob, np.uint32, (-1, 2))
        verified_data = None if verified_data_blob is None else blob_to_array(verified_data_blob, np.uint32, (-1, 2))
        # Prefer geometrically verified correspondences when they have enough support.
        data = verified_data if verified_data is not None and verified_data.shape[0] >= 30 else raw_data
        if data is None:
            invalid_count += 1
            continue
        # Convert COLMAP pair_id to image IDs
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        pair_key = (image_id1, image_id2)
        image_pairs[pair_key] = ImagePair(image_id1=image_id1, image_id2=image_id2)
        keypoints1 = images_dict[image_id1]['features']
        keypoints2 = images_dict[image_id2]['features']
        idx1 = data[:, 0]
        idx2 = data[:, 1]
        valid_indices = (idx1 != -1) & (idx2 != -1) & (idx1 < len(keypoints1)) & (idx2 < len(keypoints2))
        valid_matches = data[valid_indices]
        image_pairs[pair_key].matches = valid_matches

        config = ConfigurationType(config)
        image_pairs[pair_key].config = config
        if config in [ConfigurationType.UNDEFINED, ConfigurationType.DEGENERATE, ConfigurationType.WATERMARK, ConfigurationType.MULTIPLE]:
            image_pairs[pair_key].is_valid = False
            invalid_count += 1
            continue

        F = blob_to_array(F_blob, np.float64).reshape(3, 3) if F_blob is not None else None
        E = blob_to_array(E_blob, np.float64).reshape(3, 3) if E_blob is not None else None
        H = blob_to_array(H_blob, np.float64).reshape(3, 3) if H_blob is not None else None
        image_pairs[pair_key].F = F
        image_pairs[pair_key].E = E
        image_pairs[pair_key].H = H
        image_pairs[pair_key].config = config

    view_graph.image_pairs = {pair_key: image_pair for pair_key, image_pair in image_pairs.items() if image_pair.is_valid}
    print(f'Pairs read done. {invalid_count} / {len(image_pairs)+invalid_count} are invalid')

    # Convert dict to Images container with ID remapping
    camera_items = sorted(camera_records.items())
    cam_id2idx = {cam_id: idx for idx, (cam_id, _) in enumerate(camera_items)}
    cameras = Cameras(num_cameras=len(camera_items))
    for idx, (cam_id, cam_data) in enumerate(camera_items):
        # Camera ID is now the same as index, no need to set cameras.ids
        cameras.model_ids[idx] = cam_data['model_id'].value
        cameras.widths[idx] = cam_data['width']
        cameras.heights[idx] = cam_data['height']
        cameras.has_prior_focal_length[idx] = cam_data['has_prior_focal_length']
        cameras.set_params(idx, cam_data['params'], cam_data['model_id'])
    
    img_id2idx = {img_id:idx for idx, img_id in enumerate(images_dict.keys())}
    
    # Create Images container
    images = Images(num_images=len(images_dict))
    for idx, (img_id, image_data) in enumerate(sorted(images_dict.items())):
        # Image index is used as ID directly, no need to set ids array
        images.cam_ids[idx] = cam_id2idx[image_data['cam_id']]
        images.filenames[idx] = image_data['filename']
        images.is_registered[idx] = image_data['is_registered']
        images.cluster_ids[idx] = image_data['cluster_id']
        images.world2cams[idx] = image_data['world2cam']
        images.features[idx] = image_data['features']
        images.depths[idx] = image_data['depths']
        images.semantics[idx] = image_data['semantics']  # TODO: Load from semantics_path when available
        images.features_undist[idx] = image_data['features_undist']
        images.point3d_ids[idx] = image_data['point3d_ids']
        images.num_points3d[idx] = image_data['num_points3d']
    
    # Update image pair IDs to use the new sequential indices
    updated_pairs = {}
    for (old_id1, old_id2), pair in view_graph.image_pairs.items():
        new_id1 = img_id2idx[old_id1]
        new_id2 = img_id2idx[old_id2]
        pair.image_id1 = new_id1
        pair.image_id2 = new_id2
        updated_pairs[(new_id1, new_id2)] = pair
    view_graph.image_pairs = updated_pairs

    # assign image partners here using efficient array structure
    # detect if this is a multi-camera rig by checking if there are multiple folders with same length
    multi_camera_rig = len(image_folders) > 1 and len(set(len(folder) for folder in image_folders.values())) == 1
    
    if multi_camera_rig:
        # Build rig structure
        folder_names = sorted(image_folders.keys())
        num_groups = len(list(image_folders.values())[0])
        num_cameras = len(folder_names)
        
        # Create 2D array: rows=groups, cols=cameras
        rig_groups = np.full((num_groups, num_cameras), -1, dtype=np.int32)
        
        for group_idx in range(num_groups):
            for cam_idx, folder_name in enumerate(folder_names):
                original_id = image_folders[folder_name][group_idx]['id']
                image_id = img_id2idx[original_id]
                rig_groups[group_idx, cam_idx] = image_id
                images.image_to_rig[image_id] = [group_idx, cam_idx]
        
        images.rig_groups = rig_groups
        images.rig_folder_names = folder_names
        images.rel_poses = np.tile(np.eye(4), (num_cameras, 1, 1))
        images.ref_poses = np.tile(np.eye(4), (num_groups, 1, 1))
        
        # Try to load fixed relative poses if available
        fixed_poses_path = os.path.join(os.path.dirname(path), 'camera_relative_poses.txt')
        fixed_rel_poses, fixed_ref_camera_idx = ReadFixedRelativePoses(fixed_poses_path, folder_names)
        if fixed_rel_poses is not None:
            images.fixed_rel_poses = fixed_rel_poses
            images.rel_poses = fixed_rel_poses.copy()
            images.ref_camera_idx = fixed_ref_camera_idx

    print(f'Reading database took: {time.time() - start_time:.2f}')

    try:
        feature_name = db.execute("SELECT feature_name FROM feature_name").fetchone()[0]
    except:
        # if the database does not have feature_name, then assume it's originated from COLMAP-compatibale workflow
        feature_name = 'colmap'

    return view_graph, cameras, images, feature_name, multi_camera_rig

def ReadFixedRelativePoses(path, rig_folder_names):
    """Read fixed relative camera poses and align with rig_folder_names order.
    
    Args:
        path: Path to the relative poses file (camera_relative_poses.txt)
        rig_folder_names: List of folder names in the rig structure order
        
    Returns:
        np.ndarray: (num_positions, 4, 4) array of relative poses aligned with rig_folder_names
        int: ref_camera_idx - index of the reference camera in rig_folder_names
    """
    if not os.path.exists(path):
        return None, None
    
    # Read all camera poses from file
    camera_poses_dict = {}  # camera_name -> (camera_id, 4x4 matrix)
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Parse camera ID and name
        parts = line.split()
        if len(parts) >= 2:
            camera_id = int(parts[0])
            camera_name = parts[1]
            
            # Read 4x4 transformation matrix (next 4 lines)
            matrix = []
            for j in range(1, 5):
                if i + j < len(lines):
                    matrix_line = lines[i + j].strip()
                    if matrix_line and not matrix_line.startswith('#'):
                        values = [float(x) for x in matrix_line.split()]
                        matrix.append(values)
            
            if len(matrix) == 4:
                camera_poses_dict[camera_name] = (camera_id, np.array(matrix))
            
            i += 5  # Skip to next camera
        else:
            i += 1
    
    if not camera_poses_dict:
        return None, None
    
    # Find reference camera (camera_id == 1 or first camera)
    ref_camera_name = None
    for cam_name, (cam_id, _) in camera_poses_dict.items():
        if cam_id == 1:
            ref_camera_name = cam_name
            break
    if ref_camera_name is None:
        ref_camera_name = list(camera_poses_dict.keys())[0]
    
    # Find ref_camera_idx in rig_folder_names
    ref_camera_idx = None
    for idx, folder_name in enumerate(rig_folder_names):
        if folder_name == ref_camera_name:
            ref_camera_idx = idx
            break
    
    if ref_camera_idx is None:
        print(f'Warning: Reference camera {ref_camera_name} not found in rig_folder_names')
        ref_camera_idx = 0
    
    # Align poses with rig_folder_names order
    num_positions = len(rig_folder_names)
    rel_poses = np.tile(np.eye(4), (num_positions, 1, 1))
    
    for idx, folder_name in enumerate(rig_folder_names):
        if folder_name in camera_poses_dict:
            _, ref2rel = camera_poses_dict[folder_name]
            rel_poses[idx] = ref2rel
        else:
            print(f'Warning: No fixed pose found for camera {folder_name}, using identity')
    
    print(f'Loaded {len(camera_poses_dict)} fixed relative poses, ref_camera: {ref_camera_name} (idx={ref_camera_idx})')
    return rel_poses, ref_camera_idx

def ReadDepthsIntoFeatures(path, cameras, images):
    """Read depths and attach to image features.
    
    Args:
        path: Path to depth directory (can be root depth_vda or specific camera folder)
        cameras: Cameras object
        images: Images object
        
    Returns:
        Numpy array of all depth values (for scene scale estimation)
    """
    # Check if this is a multi-camera rig structure
    # (depth_vda has subfolders like FRONT, FRONT_LEFT, etc.)
    all_depth_values = []
    
    # Check for multi-camera structure
    # Exclude 'images' folder as it indicates single-camera mode
    subdirs = [d for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path, d)) 
               and not d.startswith('.') 
               and d != 'images']  # 'images' folder indicates single-camera mode
    has_camera_folders = any(os.path.exists(os.path.join(path, subdir, 'depths.npz')) or 
                             os.path.exists(os.path.join(path, subdir, 'npy')) 
                             for subdir in subdirs)
    
    if has_camera_folders:        
        # Cache loaded depth maps by camera folder
        camera_depth_maps = {}  # filename-based lookup
        
        for i in range(len(images)):
            image = images[i]
            filename = image.filename
            
            # Extract camera folder from filename (e.g., "FRONT/frame_000000.jpg" -> "FRONT")
            camera_folder = os.path.dirname(filename)
            
            if not camera_folder:
                # Single folder case, use image index
                camera_depths = ReadDepths(path)
                if len(camera_depths) > 0 and i < len(camera_depths):
                    depths_list = []
                    for feat in image.features:
                        depth, available = sample_depth_at_pixel(
                            camera_depths[i], feat, 
                            cameras[image.cam_id].width, 
                            cameras[image.cam_id].height
                        )
                        depths_list.append(depth)
                    images.depths[i] = np.array(depths_list, dtype=np.float32)
                    all_depth_values.extend(depths_list)
                else:
                    images.depths[i] = np.array([], dtype=np.float32)
                continue
            
            # Load depths for this camera if not already loaded
            if camera_folder not in camera_depth_maps:
                camera_depth_path = os.path.join(path, camera_folder)
                depth_map = ReadDepthsWithFilenames(camera_depth_path)
                camera_depth_maps[camera_folder] = depth_map
            
            depth_map = camera_depth_maps[camera_folder]
            # Use filename-based matching
            image_basename = os.path.splitext(os.path.basename(filename))[0]
            if image_basename in depth_map:
                depth_img = depth_map[image_basename]
                depths_list = []
                for feat in image.features:
                    depth, available = sample_depth_at_pixel(
                        depth_img, feat, 
                        cameras[image.cam_id].width, 
                        cameras[image.cam_id].height
                    )
                    depths_list.append(depth)
                images.depths[i] = np.array(depths_list, dtype=np.float32)
                all_depth_values.extend(depths_list)
            else:
                images.depths[i] = np.array([], dtype=np.float32)
    else:
        # Single camera format: depth_vda contains depths.npz or npy/ directly
        images_depth_path = os.path.join(path, 'images')
        depth_map = ReadDepthsWithFilenames(images_depth_path)

        # Use filename-based matching
        print(f"Using filename-based depth matching")
        for i in range(len(images)):
            image = images[i]
            # Get image basename without extension
            image_basename = os.path.splitext(os.path.basename(image.filename))[0]
            
            if image_basename in depth_map:
                depth_img = depth_map[image_basename]
                depths_list = []
                for feat in image.features:
                    depth, available = sample_depth_at_pixel(
                        depth_img, feat, 
                        cameras[image.cam_id].width, 
                        cameras[image.cam_id].height
                    )
                    depths_list.append(depth)
                images.depths[i] = np.array(depths_list, dtype=np.float32)
                all_depth_values.extend(depths_list)
            else:
                images.depths[i] = np.array([], dtype=np.float32)

def ReadDepthsWithFilenames(path):
    """Read depth maps and return filename-to-depth mapping.
    
    Args:
        path: Path to depth folder
        
    Returns:
        dict: Mapping from image basename (without extension) to depth map
    """
    depth_map = {}
    
    # Check for npy folder with named depth files
    npy_folder = os.path.join(path, 'npy')
    if os.path.exists(npy_folder):
        print(f"Loading named depth files from: {npy_folder}")
        depth_files = glob.glob(os.path.join(npy_folder, '*.npy'))
        if depth_files:
            for depth_file in depth_files:
                basename = os.path.basename(depth_file)
                # Remove .npy extension
                name_without_ext = os.path.splitext(basename)[0]
                depth = np.load(depth_file)
                depth_map[name_without_ext] = depth
            return depth_map
    
    return depth_map

def ReadDepths(path):
    """Read depth maps from various formats.
    
    Supports:
    - ScanNet format: PNG files with depth in millimeters
    - Video Depth Anything format: npz file or npy folder
    
    Args:
        path: Path to depth folder
    
    Returns:
        Array of depth maps [N, H, W]
    """
    # Check for Video Depth Anything format (npz file)
    npz_path = os.path.join(path, 'depths.npz')
    if os.path.exists(npz_path):
        print(f"Loading depths from npz: {npz_path}")
        data = np.load(npz_path)
        depths = data['depths']
        return depths.astype(np.float32)
    
    # Check for Video Depth Anything npy folder
    npy_folder = os.path.join(path, 'npy')
    if os.path.exists(npy_folder):
        print(f"Loading depths from npy: {npy_folder}")
        depth_files = sorted(glob.glob(os.path.join(npy_folder, 'depth_*.npy')))
        if depth_files:
            depths = []
            for depth_file in depth_files:
                depth = np.load(depth_file)
                depths.append(depth)
            depths = np.array(depths, dtype=np.float32)
            return depths
    
    # Fallback to ScanNet PNG format
    depth_files = sorted(glob.glob(os.path.join(path, '*.png')))
    print(f"Loading depths from PNG files: {path}")
    depths = []
    for depth_file in depth_files:
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # ScanNet format: depth represents millimeters
        depth = depth.astype(np.float32) / 1000.0
        depths.append(depth)
    depths = np.array(depths, dtype=np.float32)
    return depths

def ReadSemanticsIntoFeatures(path, cameras, images):
    """Read semantic segmentation masks and attach to images.
    
    Args:
        path: Path to semantics folder containing camera subfolders with masks/
        cameras: Cameras object
        images: Images object
    
    Returns:
        Number of images with loaded semantics
    """    
    semantics_loaded = 0
    
    for i in range(len(images)):
        image = images[i]
        filename = image.filename
        
        # Extract frame number from filename (e.g., "FRONT_LEFT/frame_000000_ts_xxx.jpg" -> "000000")
        basename = os.path.basename(filename)
        if basename.startswith('frame_'):
            # Extract frame number: frame_000000_ts_xxx.jpg -> 000000
            frame_num = basename.split('_')[1]
            
            # Get camera folder from filename path
            camera_folder = os.path.dirname(filename)
            
            # Construct path to semantic mask
            mask_path = os.path.join(path, camera_folder, 'masks', f'frame_{frame_num}.npy')
            
            if os.path.exists(mask_path):
                # Load semantic mask (N_objects, H, W) binary masks
                masks = np.load(mask_path)
                
                if len(masks) == 0:
                    # No objects detected in this frame
                    images.semantics[i] = None
                    continue
                
                # Convert binary masks to instance ID map (H, W)
                # Each pixel gets the ID of the first object mask it belongs to
                h, w = masks.shape[1], masks.shape[2]
                instance_map = np.zeros((h, w), dtype=np.int32)
                
                for obj_idx, mask in enumerate(masks):
                    # Object IDs start from 1 (0 is background)
                    instance_map[mask > 0] = obj_idx + 1
                
                images.semantics[i] = instance_map
                semantics_loaded += 1
            else:
                images.semantics[i] = None
        else:
            images.semantics[i] = None
    
    print(f"Loaded semantics for {semantics_loaded}/{len(images)} images")
    return semantics_loaded
