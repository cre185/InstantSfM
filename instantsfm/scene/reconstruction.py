import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

from instantsfm.scene.defs import get_camera_model_info
from instantsfm.utils.read_write_model import write_next_bytes

def bilinear_interpolate(image, x, y):
    """Vectorized bilinear interpolation for batch of points.
    
    Args:
        image: (H, W, C) image array
        x: (N,) array of x coordinates
        y: (N,) array of y coordinates
        
    Returns:
        (N, C) array of interpolated colors
    """
    h, w, c = image.shape
    
    # Clamp coordinates
    x = np.clip(x, 0., w - 1.)
    y = np.clip(y, 0., h - 1.)
    
    # Get integer parts
    x1 = np.floor(x).astype(np.int32)
    y1 = np.floor(y).astype(np.int32)
    x2 = np.clip(x1 + 1, 0, w - 1)
    y2 = np.clip(y1 + 1, 0, h - 1)
    
    # Get fractional parts
    fx = x - x1
    fy = y - y1
    
    # Bilinear interpolation weights
    w11 = ((1 - fx) * (1 - fy))[:, np.newaxis]  # (N, 1)
    w12 = ((1 - fx) * fy)[:, np.newaxis]
    w21 = (fx * (1 - fy))[:, np.newaxis]
    w22 = (fx * fy)[:, np.newaxis]
    
    # Interpolate
    colors = (w11 * image[y1, x1] + 
              w12 * image[y2, x1] + 
              w21 * image[y1, x2] + 
              w22 * image[y2, x2])
    
    return colors

class point3d:
    def __init__(self, **kwargs):
        self.xyz = np.zeros(3)
        self.color = np.zeros(3)
        self.error = -1.
        self.track_elements = [] # track_element = (image_id, point2d_idx)
        for key, val in kwargs.items():
            setattr(self, key, val)

class Reconstruction:
    def __init__(self, cameras, images, tracks):
        """Initialize reconstruction with cameras, Images container, and Tracks container.
        
        Args:
            cameras: List of camera models
            images: Images container (optional)
            tracks: Tracks container (optional)
        """
        self.cameras = cameras
        self.images = images  # Images container
        self.tracks = tracks  # Tracks container
        self._selected_indices = None  # Filtered image indices
        self._point3d_ids = None  # 2D-3D correspondences
    
    @property
    def num_images(self):
        """Number of selected images."""
        if self._selected_indices is not None:
            return len(self._selected_indices)
        if self.images is not None:
            return np.sum(self.images.is_registered)
        return 0
    
    @property
    def num_points(self):
        """Number of tracks."""
        if self.tracks is not None:
            return len(self.tracks)
        return 0
    
    def filter_by_cluster(self, cluster_id):
        """Filter images by cluster ID."""
        if not hasattr(self.images, 'cluster_ids'):
            return
        
        if self._selected_indices is None:
            self._selected_indices = np.arange(len(self.images))
        
        mask = self.images.cluster_ids[self._selected_indices] == cluster_id
        self._selected_indices = self._selected_indices[mask]
    
    def filter_registered_only(self):
        """Filter to keep only registered images."""
        if self.images is None:
            return
        
        if self._selected_indices is None:
            self._selected_indices = np.where(self.images.is_registered)[0]
        else:
            mask = self.images.is_registered[self._selected_indices]
            self._selected_indices = self._selected_indices[mask]
    
    def build_correspondences(self, min_track_length=2):
        """Build 2D-3D correspondences using batch operations.
        
        Args:
            min_track_length: Minimum number of observations per track
        """
        if self.images is None or self.tracks is None:
            return
        
        if self._selected_indices is None:
            self._selected_indices = np.where(self.images.is_registered)[0]
        
        # Initialize point3d_ids for all images
        self._point3d_ids = [None] * len(self.images)
        for idx in self._selected_indices:
            num_features = len(self.images.features[idx])
            self._point3d_ids[idx] = -np.ones(num_features, dtype=np.int32)
        
        # Batch build correspondences
        valid_tracks = 0
        for track_id in range(len(self.tracks)):
            obs = self.tracks.observations[track_id]
            if len(obs) < min_track_length:
                continue
            
            valid_tracks += 1
            # Assign track_id to all observations
            for img_idx, feat_idx in obs:
                if self._point3d_ids[img_idx] is not None:
                    self._point3d_ids[img_idx][feat_idx] = track_id
        
        print(f'Built correspondences for {valid_tracks} tracks')
    
    def extract_colors_batch(self, image_path):
        """Extract colors for all tracks using batch processing.
        
        Args:
            image_path: Path to image directory
        """        
        # Accumulate colors per track
        color_sums = np.zeros((len(self.tracks), 3), dtype=np.float64)
        color_counts = np.zeros(len(self.tracks), dtype=np.int32)
        
        # Process each image
        for idx in self._selected_indices:
            filename = self.images.filenames[idx]
            if filename is None or not os.path.exists(os.path.join(image_path, filename)):
                continue
            
            # Load image
            bitmap = cv2.imread(os.path.join(image_path, filename))
            bitmap = cv2.cvtColor(bitmap, cv2.COLOR_BGR2RGB)
            
            # Get valid correspondences for this image
            point3d_ids = self._point3d_ids[idx]
            valid_mask = point3d_ids != -1
            if not np.any(valid_mask):
                continue
            
            # Batch interpolate colors - fully vectorized
            features = self.images.features[idx][valid_mask]  # (N, 2)
            track_ids = point3d_ids[valid_mask]  # (N,)
            
            # Extract coordinates and interpolate colors in batch
            xy = features - 0.5  # (N, 2)
            colors = bilinear_interpolate(bitmap, xy[:, 0], xy[:, 1])  # (N, 3)
            
            # Accumulate colors per track using vectorized operations
            np.add.at(color_sums, track_ids, colors)
            np.add.at(color_counts, track_ids, 1)
        
        # Compute average colors
        valid_mask = color_counts > 0
        self.tracks.colors[valid_mask] = (color_sums[valid_mask] / color_counts[valid_mask, None]).astype(np.uint8)
        
        print(f'Extracted colors for {np.sum(valid_mask)} / {len(self.tracks)} tracks')
    
    def write_binary(self, path):
        """Write reconstruction in COLMAP binary format using batch operations."""
        self._write_cameras_binary(os.path.join(path, 'cameras.bin'))
        self._write_images_binary(os.path.join(path, 'images.bin'))
        self._write_points3d_binary(os.path.join(path, 'points3D.bin'))
    
    def write_text(self, path):
        """Write reconstruction in COLMAP text format using batch operations."""
        self._write_cameras_text(os.path.join(path, 'cameras.txt'))
        self._write_images_text(os.path.join(path, 'images.txt'))
        self._write_points3d_text(os.path.join(path, 'points3D.txt'))
    
    def _write_cameras_binary(self, filepath):
        """Write cameras in binary format."""
        with open(filepath, "wb") as fid:
            cameras_list = list(self.cameras.values()) if isinstance(self.cameras, dict) else self.cameras
            write_next_bytes(fid, len(cameras_list), "Q")
            for idx, cam in enumerate(cameras_list):
                model_id = cam.model_id.value
                # Camera ID is the same as index
                camera_properties = [idx, model_id, cam.width, cam.height]
                write_next_bytes(fid, camera_properties, "iiQQ")
                for p in cam.params:
                    write_next_bytes(fid, float(p), "d")
    
    def _write_images_binary(self, filepath):
        """Write images in binary format using batch operations."""
        if self.images is None or self._selected_indices is None:
            return
        
        with open(filepath, "wb") as fid:
            write_next_bytes(fid, len(self._selected_indices), "Q")
            
            # Batch extract rotation and translation
            for idx in self._selected_indices:
                img_id = idx  # Use index as ID directly
                world2cam = self.images.world2cams[idx]
                
                # Extract pose
                tvec = world2cam[:3, 3]
                qvec = R.from_matrix(world2cam[:3, :3]).as_quat()  # xyzw
                qvec_colmap = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])  # wxyz
                
                # Write image header
                write_next_bytes(fid, img_id, "i")
                write_next_bytes(fid, qvec_colmap.tolist(), "dddd")
                write_next_bytes(fid, tvec.tolist(), "ddd")
                # Camera ID is the index itself
                write_next_bytes(fid, int(self.images.cam_ids[idx]), "i")
                
                # Write filename
                filename = self.images.filenames[idx] if hasattr(self.images, 'filenames') else f"{img_id}.jpg"
                for char in filename:
                    write_next_bytes(fid, char.encode("utf-8"), "c")
                write_next_bytes(fid, b"\x00", "c")
                
                # Write 2D points and 3D correspondences
                point3d_ids = self._point3d_ids[idx]
                valid_mask = point3d_ids != -1
                features = self.images.features[idx][valid_mask]
                valid_point3d_ids = point3d_ids[valid_mask]
                
                write_next_bytes(fid, len(valid_point3d_ids), "Q")
                for xy, p3d_id in zip(features, valid_point3d_ids):
                    write_next_bytes(fid, [float(xy[0]), float(xy[1]), int(p3d_id)], "ddq")
    
    def _write_points3d_binary(self, filepath):
        """Write 3D points in binary format using batch operations."""
        if self.tracks is None:
            return
        
        with open(filepath, "wb") as fid:
            write_next_bytes(fid, len(self.tracks), "Q")
            
            for track_id in range(len(self.tracks)):
                xyz = self.tracks.xyzs[track_id]
                color = self.tracks.colors[track_id]
                obs = self.tracks.observations[track_id]
                
                write_next_bytes(fid, track_id, "Q")
                write_next_bytes(fid, xyz.tolist(), "ddd")
                write_next_bytes(fid, [int(c) for c in color], "BBB")
                write_next_bytes(fid, 0.0, "d")  # error
                write_next_bytes(fid, len(obs), "Q")
                
                for image_id, point2d_id in obs:
                    write_next_bytes(fid, [int(image_id), int(point2d_id)], "ii")
    
    def _write_cameras_text(self, filepath):
        """Write cameras in text format."""
        cameras_list = list(self.cameras.values()) if isinstance(self.cameras, dict) else self.cameras
        HEADER = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(cameras_list)}\n"
        )
        with open(filepath, "w") as fid:
            fid.write(HEADER)
            for idx, cam in enumerate(cameras_list):
                model = get_camera_model_info(cam.model_id)["name"]
                # Camera ID is the same as index
                to_write = [idx, model, cam.width, cam.height, *cam.params]
                line = " ".join([str(elem) for elem in to_write])
                fid.write(line + "\n")
    
    def _write_images_text(self, filepath):
        """Write images in text format using batch operations."""
        if self.images is None or self._selected_indices is None:
            return
        
        # Calculate mean observations
        mean_observations = 0
        if len(self._selected_indices) > 0:
            total_obs = sum(np.sum(self._point3d_ids[idx] != -1) for idx in self._selected_indices)
            mean_observations = total_obs / len(self._selected_indices)
        
        HEADER = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(self._selected_indices)}, mean observations per image: {mean_observations}\n"
        )
        
        with open(filepath, "w") as fid:
            fid.write(HEADER)
            
            for idx in self._selected_indices:
                img_id = idx  # Use index as ID directly
                world2cam = self.images.world2cams[idx]
                
                # Extract pose
                tvec = world2cam[:3, 3]
                qvec = R.from_matrix(world2cam[:3, :3]).as_quat()  # xyzw
                qvec_colmap = [qvec[3], qvec[0], qvec[1], qvec[2]]  # wxyz
                
                filename = self.images.filenames[idx] if hasattr(self.images, 'filenames') else f"{img_id}.jpg"
                image_header = [img_id, *qvec_colmap, *tvec, int(self.images.cam_ids[idx]), filename]
                first_line = " ".join(map(str, image_header))
                fid.write(first_line + "\n")
                
                # Write 2D points
                features = self.images.features[idx]
                point3d_ids = self._point3d_ids[idx]
                
                points_strings = []
                for xy, p3d_id in zip(features, point3d_ids):
                    points_strings.append(" ".join(map(str, [xy[0], xy[1], int(p3d_id)])))
                fid.write(" ".join(points_strings) + "\n")
    
    def _write_points3d_text(self, filepath):
        """Write 3D points in text format using batch operations."""
        if self.tracks is None:
            return
        
        # Calculate mean track length
        mean_track_length = 0
        if len(self.tracks) > 0:
            total_obs = sum(len(self.tracks.observations[i]) for i in range(len(self.tracks)))
            mean_track_length = total_obs / len(self.tracks)
        
        HEADER = (
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, SEMANTIC_LABEL, SEMANTIC_CONF, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {len(self.tracks)}, mean track length: {mean_track_length}\n"
        )
        
        with open(filepath, "w") as fid:
            fid.write(HEADER)
            
            for track_id in range(len(self.tracks)):
                xyz = self.tracks.xyzs[track_id]
                color = self.tracks.colors[track_id]
                obs = self.tracks.observations[track_id]
                semantic_label = int(self.tracks.semantic_labels[track_id])
                semantic_conf = float(self.tracks.semantic_confidences[track_id])
                
                point_header = [track_id, *xyz.tolist(), *[int(c) for c in color], 0.0, semantic_label, semantic_conf]
                fid.write(" ".join(map(str, point_header)) + " ")
                
                track_strings = []
                for image_id, point2d_id in obs:
                    track_strings.append(" ".join(map(str, [int(image_id), int(point2d_id)])))
                fid.write(" ".join(track_strings) + "\n")