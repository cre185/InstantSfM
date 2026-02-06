from queue import Queue
import numpy as np
from enum import Enum
import cv2
from typing import List, Optional, Dict, Tuple, Union
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Struct-of-Arrays (SoA) Image Container
# ============================================================================

class ImageView:
    """A lightweight view of a single image in the Images container.
    
    Provides dict-like access to image attributes without copying data.
    Changes to this view are reflected in the underlying container.
    """
    def __init__(self, container: 'Images', index: int):
        self._container = container
        self._index = index
    
    @property
    def id(self) -> int:
        """Return the image index (used as ID)."""
        return self._index
    
    @property
    def cam_id(self) -> int:
        return int(self._container.cam_ids[self._index])
    
    @property
    def filename(self) -> str:
        return self._container.filenames[self._index]
    
    @property
    def is_registered(self) -> bool:
        return bool(self._container.is_registered[self._index])
    
    @is_registered.setter
    def is_registered(self, value: bool):
        self._container.is_registered[self._index] = value
    
    @property
    def cluster_id(self) -> int:
        return int(self._container.cluster_ids[self._index])
    
    @cluster_id.setter
    def cluster_id(self, value: int):
        self._container.cluster_ids[self._index] = value
    
    @property
    def world2cam(self) -> np.ndarray:
        return self._container.world2cams[self._index]
    
    @world2cam.setter
    def world2cam(self, value: np.ndarray):
        self._container.world2cams[self._index] = value
    
    @property
    def features(self) -> np.ndarray:
        return self._container.features[self._index]
    
    @property
    def depths(self) -> np.ndarray:
        return self._container.depths[self._index]
    
    @property
    def semantics(self) -> Optional[np.ndarray]:
        return self._container.semantics[self._index]
    
    @semantics.setter
    def semantics(self, value: Optional[np.ndarray]):
        self._container.semantics[self._index] = value
    
    @property
    def features_undist(self) -> np.ndarray:
        return self._container.features_undist[self._index]
    
    @property
    def point3d_ids(self) -> List[int]:
        return self._container.point3d_ids[self._index]
    
    @property
    def num_points3d(self) -> int:
        return int(self._container.num_points3d[self._index])
    
    @num_points3d.setter
    def num_points3d(self, value: int):
        self._container.num_points3d[self._index] = value
    
    @property
    def rig_info(self) -> tuple:
        """Get rig information for this image.
        
        Returns:
            (group_idx, camera_idx) or (-1, -1) if not part of a rig
        """
        return tuple(self._container.image_to_rig[self._index])
    
    @property
    def partner_ids(self) -> Dict:
        """Get partner image IDs as dict (for backward compatibility).
        
        Returns empty dict if not part of a rig.
        """
        group_idx, cam_idx = self.rig_info
        if group_idx == -1:
            return {}
        
        # Build dict from rig_groups array
        result = {}
        rig_row = self._container.rig_groups[group_idx]
        for i, folder_name in enumerate(self._container.rig_folder_names):
            if rig_row[i] != -1:
                result[folder_name] = rig_row[i]
        return result
    
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        w2c = self.world2cam
        return w2c[:3, :3].T @ -w2c[:3, 3]
    
    def axis_angle(self) -> np.ndarray:
        """Rotation as axis-angle representation."""
        return R.from_matrix(self.world2cam[:3, :3]).as_rotvec()


class Images:
    """Struct-of-Arrays container for efficient batched image operations.
    
    Stores all image data contiguously for GPU-friendly operations.
    Supports both batched operations and per-image indexing.
    
    Usage:
        images = Images(num_images=100)
        images.world2cams[0] = np.eye(4)  # Direct batched access
        img = images[0]  # Get ImageView for single image
        print(img.filename)  # Access through view
    """
    
    def __init__(self, num_images: int = 0):
        self.num_images = num_images
        
        # Scalar attributes (N,)
        # Note: We use array index as image ID directly, no separate ids array needed
        self.cam_ids = np.full(num_images, -1, dtype=np.int32)
        self.is_registered = np.zeros(num_images, dtype=bool)
        self.cluster_ids = np.full(num_images, -1, dtype=np.int32)
        self.num_points3d = np.zeros(num_images, dtype=np.int32)
        
        # String attributes (list of strings)
        self.filenames = [""] * num_images
        
        # Matrix attributes (N, 4, 4)
        self.world2cams = np.tile(np.eye(4), (num_images, 1, 1))
        
        # Variable-length arrays (list of arrays)
        self.features = [np.array([]) for _ in range(num_images)]
        self.features_undist = [np.array([]) for _ in range(num_images)]
        self.point3d_ids = [[] for _ in range(num_images)]

        # Depths
        self.depths = [np.array([]) for _ in range(num_images)]

        # Semantics
        self.semantics = [None for _ in range(num_images)]  # Per-pixel instance IDs (H, W) int array or None
        
        # Multi-camera rig structure
        self.rig_groups = np.array([]).reshape(0, 0)  # Empty initially
        self.rig_folder_names = []  # Folder names for each camera position
        self.image_to_rig = np.full((num_images, 2), -1, dtype=np.int32)  # (group_idx, camera_idx)
        self.fixed_rel_poses = None  # (num_positions, 4, 4) Fixed relative poses (ref_to_camera) if available
        
        # Rig pose decomposition
        self.ref_camera_idx = None  # Index of the reference camera in the rig
        self.ref_poses = None  # (num_groups, 4, 4) Reference camera poses (world2cam) for each rig group
        self.rel_poses = None  # (num_positions, 4, 4) Relative poses from reference to each camera position
        
        # Rig voting results
        self.image_votes = None  # (num_images,) Vote weights for each image used in relative pose voting
    
    def __len__(self) -> int:
        return self.num_images
    
    def __getitem__(self, index: int) -> ImageView:
        """Get a view of a single image."""
        if index < 0 or index >= self.num_images:
            raise IndexError(f"Image index {index} out of range [0, {self.num_images})")
        return ImageView(self, index)
    
    def get_registered_mask(self) -> np.ndarray:
        """Get boolean mask of registered images."""
        return self.is_registered
    
    def get_registered_indices(self) -> np.ndarray:
        """Get indices of registered images."""
        return np.where(self.is_registered)[0]
    
    def get_world2cams_batch(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get world2cam matrices as a batch.
        
        Args:
            indices: Optional array of image indices. If None, returns all.
            
        Returns:
            Array of shape (N, 4, 4) or (len(indices), 4, 4).
        """
        if indices is None:
            return self.world2cams
        return self.world2cams[indices]
    
    def get_centers_batch(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get camera centers in world coordinates as a batch.
        
        Args:
            indices: Optional array of image indices. If None, returns all.
            
        Returns:
            Array of shape (N, 3) or (len(indices), 3).
        """
        w2cs = self.get_world2cams_batch(indices)
        R = w2cs[:, :3, :3]
        t = w2cs[:, :3, 3]
        # centers = -R^T @ t
        return np.einsum('nij,nj->ni', R.transpose(0, 2, 1), -t)
    
    def get_rotations_batch(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get rotation matrices as a batch.
        
        Args:
            indices: Optional array of image indices. If None, returns all.
            
        Returns:
            Array of shape (N, 3, 3) or (len(indices), 3, 3).
        """
        w2cs = self.get_world2cams_batch(indices)
        return w2cs[:, :3, :3]
    
    def set_world2cams_batch(self, world2cams: np.ndarray, indices: Optional[np.ndarray] = None):
        """Set world2cam matrices from a batch.
        
        Args:
            world2cams: Array of shape (N, 4, 4) or (len(indices), 4, 4).
            indices: Optional array of image indices. If None, sets all.
        """
        if indices is None:
            assert world2cams.shape[0] == self.num_images
            self.world2cams[:] = world2cams
        else:
            assert world2cams.shape[0] == len(indices)
            self.world2cams[indices] = world2cams


# ============================================================================
# Struct-of-Arrays (SoA) Track Container
# ============================================================================

class TrackView:
    """A lightweight view of a single track in the Tracks container.
    
    Provides dict-like access to track attributes without copying data.
    Changes to this view are reflected in the underlying container.
    """
    def __init__(self, container: 'Tracks', index: int):
        self._container = container
        self._index = index
    
    @property
    def id(self) -> int:
        return int(self._container.ids[self._index])
    
    @property
    def xyz(self) -> np.ndarray:
        return self._container.xyzs[self._index]
    
    @xyz.setter
    def xyz(self, value: np.ndarray):
        self._container.xyzs[self._index] = value
    
    @property
    def color(self) -> np.ndarray:
        return self._container.colors[self._index]
    
    @color.setter
    def color(self, value: np.ndarray):
        self._container.colors[self._index] = value
    
    @property
    def semantic_label(self) -> int:
        return int(self._container.semantic_labels[self._index])
    
    @semantic_label.setter
    def semantic_label(self, value: int):
        self._container.semantic_labels[self._index] = value
    
    @property
    def semantic_confidence(self) -> float:
        return float(self._container.semantic_confidences[self._index])
    
    @semantic_confidence.setter
    def semantic_confidence(self, value: float):
        self._container.semantic_confidences[self._index] = value
    
    @property
    def is_initialized(self) -> bool:
        return bool(self._container.is_initialized[self._index])
    
    @is_initialized.setter
    def is_initialized(self, value: bool):
        self._container.is_initialized[self._index] = value
    
    @property
    def observations(self) -> np.ndarray:
        return self._container.observations[self._index]
    
    @observations.setter
    def observations(self, value: np.ndarray):
        self._container.observations[self._index] = value


class Tracks:
    """Struct-of-Arrays container for efficient batched track operations.
    
    Stores all track data contiguously for GPU-friendly operations.
    Supports both batched operations and per-track indexing.
    
    Usage:
        tracks = Tracks(num_tracks=1000)
        tracks.xyzs[:10] = np.random.randn(10, 3)  # Direct batched access
        track = tracks[0]  # Get TrackView for single track
        print(track.xyz)  # Access through view
    """
    
    def __init__(self, num_tracks: int = 0):
        self.num_tracks = num_tracks
        
        # Scalar attributes
        self.ids = np.full(num_tracks, -1, dtype=np.int32)
        self.is_initialized = np.zeros(num_tracks, dtype=bool)
        
        # 3D positions (N, 3)
        self.xyzs = np.zeros((num_tracks, 3), dtype=np.float64)
        
        # Colors (N, 3)
        self.colors = np.zeros((num_tracks, 3), dtype=np.uint8)
        
        # Semantic labels (N,) - dominant instance ID per track, -1 if unknown
        self.semantic_labels = np.full(num_tracks, -1, dtype=np.int32)
        
        # Semantic confidences (N,) - proportion of observations agreeing with dominant label
        self.semantic_confidences = np.zeros(num_tracks, dtype=np.float32)
        
        # Variable-length observations (list of arrays)
        # Each observation is array of shape (num_obs, 2) containing (image_id, feature_id)
        self.observations = [np.zeros((0, 2), dtype=np.int32) for _ in range(num_tracks)]
    
    def __len__(self) -> int:
        return self.num_tracks
    
    def __getitem__(self, index: int) -> TrackView:
        """Get a view of a single track."""
        if index < 0 or index >= self.num_tracks:
            raise IndexError(f"Track index {index} out of range [0, {self.num_tracks})")
        return TrackView(self, index)
    
    def append(self,
               id: int = -1,
               xyz: Optional[np.ndarray] = None,
               color: Optional[np.ndarray] = None,
               is_initialized: bool = False,
               observations: Optional[np.ndarray] = None,
               semantic_label: int = -1,
               semantic_confidence: float = 0.0) -> int:
        """Append a new track to the container.
        
        Returns:
            Index of the newly added track.
        """
        idx = self.num_tracks
        self.num_tracks += 1
        
        # Resize arrays
        self.ids = np.append(self.ids, id)
        self.is_initialized = np.append(self.is_initialized, is_initialized)
        
        xyz_val = xyz if xyz is not None else np.zeros(3)
        self.xyzs = np.vstack([self.xyzs, xyz_val.reshape(1, 3)])
        
        color_val = color if color is not None else np.zeros(3, dtype=np.uint8)
        self.colors = np.vstack([self.colors, color_val.reshape(1, 3)])
        
        self.semantic_labels = np.append(self.semantic_labels, semantic_label)
        self.semantic_confidences = np.append(self.semantic_confidences, semantic_confidence)
        
        obs_val = observations if observations is not None else np.zeros((0, 2), dtype=np.int32)
        self.observations.append(obs_val)
        
        return idx
    
    def get_initialized_mask(self) -> np.ndarray:
        """Get boolean mask of initialized tracks."""
        return self.is_initialized
    
    def get_initialized_indices(self) -> np.ndarray:
        """Get indices of initialized tracks."""
        return np.where(self.is_initialized)[0]
    
    def get_xyzs_batch(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get 3D positions as a batch.
        
        Args:
            indices: Optional array of track indices. If None, returns all.
            
        Returns:
            Array of shape (N, 3) or (len(indices), 3).
        """
        if indices is None:
            return self.xyzs
        return self.xyzs[indices]
    
    def set_xyzs_batch(self, xyzs: np.ndarray, indices: Optional[np.ndarray] = None):
        """Set 3D positions from a batch.
        
        Args:
            xyzs: Array of shape (N, 3) or (len(indices), 3).
            indices: Optional array of track indices. If None, sets all.
        """
        if indices is None:
            assert xyzs.shape[0] == self.num_tracks
            self.xyzs[:] = xyzs
        else:
            assert xyzs.shape[0] == len(indices)
            self.xyzs[indices] = xyzs
    
    def filter_by_mask(self, mask: np.ndarray) -> int:
        """Filter tracks in-place by boolean mask or indices.
        
        This method modifies the container in-place, keeping only tracks
        where mask is True. This is essential for maintaining reference
        semantics when the container is passed by reference.
        
        Args:
            mask: Boolean array of shape (num_tracks,) or integer indices
            
        Returns:
            Number of tracks after filtering
            
        Example:
            >>> tracks = Tracks(num_tracks=100)
            >>> mask = tracks.is_initialized
            >>> tracks.filter_by_mask(mask)  # Keep only initialized tracks
        """
        # Convert to boolean mask if indices are provided
        if mask.dtype != bool:
            valid_indices = mask
            mask = np.zeros(len(self), dtype=bool)
            mask[valid_indices] = True
        
        valid_indices = np.where(mask)[0]
        
        # Save old data
        old_ids = self.ids.copy()
        old_xyzs = self.xyzs.copy()
        old_colors = self.colors.copy()
        old_is_initialized = self.is_initialized.copy()
        old_observations = [obs.copy() if obs is not None else None for obs in self.observations]
        
        # Resize arrays in-place
        self.num_tracks = len(valid_indices)
        self.ids = np.zeros(self.num_tracks, dtype=np.int32)
        self.xyzs = np.zeros((self.num_tracks, 3), dtype=np.float64)
        self.colors = np.zeros((self.num_tracks, 3), dtype=np.uint8)
        self.is_initialized = np.zeros(self.num_tracks, dtype=bool)
        self.observations = []
        
        # Copy valid data
        for new_idx, old_idx in enumerate(valid_indices):
            self.ids[new_idx] = old_ids[old_idx]
            self.xyzs[new_idx] = old_xyzs[old_idx]
            self.colors[new_idx] = old_colors[old_idx]
            self.is_initialized[new_idx] = old_is_initialized[old_idx]
            self.observations.append(old_observations[old_idx])
        
        return self.num_tracks
    
    def get_colors_batch(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get colors as a batch.
        
        Args:
            indices: Optional array of track indices. If None, returns all.
            
        Returns:
            Array of shape (N, 3) or (len(indices), 3).
        """
        if indices is None:
            return self.colors
        return self.colors[indices]
    
    def flatten_observations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten observations into batched arrays.
        
        Returns:
            track_indices: Array of shape (total_obs,) with track index for each observation
            image_ids: Array of shape (total_obs,) with image ID for each observation
            feature_ids: Array of shape (total_obs,) with feature ID for each observation
        """
        track_indices = []
        image_ids = []
        feature_ids = []
        
        for track_idx in range(self.num_tracks):
            obs = self.observations[track_idx]
            if len(obs) > 0:
                track_indices.extend([track_idx] * len(obs))
                image_ids.extend(obs[:, 0])
                feature_ids.extend(obs[:, 1])
        
        return (np.array(track_indices, dtype=np.int32),
                np.array(image_ids, dtype=np.int32),
                np.array(feature_ids, dtype=np.int32))


# ============================================================================
# Configuration Types
# ============================================================================

class ConfigurationType(Enum):
    UNDEFINED = 0
    DEGENERATE = 1
    CALIBRATED = 2
    UNCALIBRATED = 3
    PLANAR = 4
    PANORAMIC = 5
    PLANAR_OR_PANORAMIC = 6
    WATERMARK = 7
    MULTIPLE = 8


# ============================================================================
# Image Pair
# ============================================================================
    
class ImagePair:
    def __init__(
        self,
        image_id1: int = -1,
        image_id2: int = -1,
        is_valid: bool = True,
        weight: float = 0.0,
        E: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        inliers: Optional[List] = None,
        config: ConfigurationType = ConfigurationType.UNDEFINED
    ):
        self.image_id1 = image_id1
        self.image_id2 = image_id2
        self.is_valid = is_valid
        self.weight = weight
        self.E = E if E is not None else np.eye(3)
        self.F = F if F is not None else np.eye(3)
        self.H = H if H is not None else np.eye(3)
        # below are in the form of image 1 to image 2
        self.rotation = rotation if rotation is not None else np.array([1, 0, 0, 0])
        self.translation = translation if translation is not None else np.zeros(3)
        self.inliers = inliers if inliers is not None else []
        self.config = config

    def set_cam1to2(self, cam1to2: np.ndarray) -> None:
        rotation_matrix = cam1to2[:3, :3]
        self.rotation = R.from_matrix(rotation_matrix).as_quat(canonical=False)
        self.translation = cam1to2[:3, 3]
    
    def get_cam1to2(self) -> np.ndarray:
        rotation_matrix = R.from_quat(self.rotation).as_matrix()
        return np.vstack([np.hstack([rotation_matrix, self.translation[:, np.newaxis]]), [0, 0, 0, 1]])


# ============================================================================
# Camera Models
# ============================================================================

class CameraModelId(Enum):
    INVALID = -1
    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3
    OPENCV = 4
    OPENCV_FISHEYE = 5
    FULL_OPENCV = 6
    FOV = 7
    SIMPLE_RADIAL_FISHEYE = 8
    RADIAL_FISHEYE = 9
    THIN_PRISM_FISHEYE = 10

def get_camera_model_info(model_id):
    """Get camera model information including parameter structure."""    
    if model_id == CameraModelId.SIMPLE_PINHOLE:
        return {'name': 'SIMPLE_PINHOLE', 'num_params': 3, 'focal': [0], 'pp': [1, 2], 'k': [], 'p': [], 'omega': [], 'sx': [], 'optimize': [0]}
    elif model_id == CameraModelId.PINHOLE:
        return {'name': 'PINHOLE', 'num_params': 4, 'focal': [0, 1], 'pp': [2, 3], 'k': [], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 1]}
    elif model_id == CameraModelId.SIMPLE_RADIAL:
        return {'name': 'SIMPLE_RADIAL', 'num_params': 4, 'focal': [0], 'pp': [1, 2], 'k': [3], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3]}
    elif model_id == CameraModelId.RADIAL:
        return {'name': 'RADIAL', 'num_params': 5, 'focal': [0], 'pp': [1, 2], 'k': [3, 4], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3, 4]}
    elif model_id == CameraModelId.OPENCV:
        return {'name': 'OPENCV', 'num_params': 8, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5], 'p': [6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7]}
    elif model_id == CameraModelId.OPENCV_FISHEYE:
        return {'name': 'OPENCV_FISHEYE', 'num_params': 8, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7]}
    elif model_id == CameraModelId.FULL_OPENCV:
        return {'name': 'FULL_OPENCV', 'num_params': 12, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 8, 9, 10, 11], 'p': [6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]}
    elif model_id == CameraModelId.FOV:
        return {'name': 'FOV', 'num_params': 5, 'focal': [0, 1], 'pp': [2, 3], 'k': [], 'p': [], 'omega': [4], 'sx': [], 'optimize': [0, 1, 4]}
    elif model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
        return {'name': 'SIMPLE_RADIAL_FISHEYE', 'num_params': 4, 'focal': [0], 'pp': [1, 2], 'k': [3], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3]}
    elif model_id == CameraModelId.RADIAL_FISHEYE:
        return {'name': 'RADIAL_FISHEYE', 'num_params': 5, 'focal': [0], 'pp': [1, 2], 'k': [3, 4], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3, 4]}
    elif model_id == CameraModelId.THIN_PRISM_FISHEYE:
        return {'name': 'THIN_PRISM_FISHEYE', 'num_params': 12, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 8, 9], 'p': [6, 7], 'omega': [], 'sx': [10, 11], 'optimize': [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]}
    else:
        raise NotImplementedError

class CameraView:
    """Lightweight view over Cameras container for backward compatibility."""

    def __init__(self, container: 'Cameras', index: int):
        self._container = container
        self._index = index

    def _k_values(self) -> np.ndarray:
        count = self._container.k_counts[self._index]
        if count == 0:
            return np.zeros(1, dtype=np.float64)
        return self._container.k_params[self._index, :count]

    @property
    def id(self) -> int:
        # Camera ID is the same as index
        return self._index

    @property
    def model_id(self) -> CameraModelId:
        return CameraModelId(int(self._container.model_ids[self._index]))

    @property
    def model(self) -> str:
        return get_camera_model_info(self.model_id)['name']

    @property
    def width(self) -> int:
        return int(self._container.widths[self._index])

    @width.setter
    def width(self, value: int):
        self._container.widths[self._index] = value

    @property
    def height(self) -> int:
        return int(self._container.heights[self._index])

    @height.setter
    def height(self, value: int):
        self._container.heights[self._index] = value

    @property
    def has_prior_focal_length(self) -> bool:
        return bool(self._container.has_prior_focal_length[self._index])

    @has_prior_focal_length.setter
    def has_prior_focal_length(self, value: bool):
        self._container.has_prior_focal_length[self._index] = value

    @property
    def params(self) -> np.ndarray:
        count = self._container.param_sizes[self._index]
        return self._container.params[self._index, :count]

    @params.setter
    def params(self, value: np.ndarray):
        self._container.set_params(self._index, value)

    def set_params(self, params: np.ndarray):
        self._container.set_params(self._index, params)

    @property
    def focal_length(self) -> np.ndarray:
        return self._container.focal_lengths[self._index]

    @focal_length.setter
    def focal_length(self, value: np.ndarray):
        self._container.focal_lengths[self._index] = value

    @property
    def principal_point(self) -> np.ndarray:
        return self._container.principal_points[self._index]

    @principal_point.setter
    def principal_point(self, value: np.ndarray):
        self._container.principal_points[self._index] = value

    @property
    def k(self) -> np.ndarray:
        return self._k_values()

    @property
    def p(self) -> np.ndarray:
        return self._container.p_params[self._index]

    @property
    def omega(self) -> float:
        return float(self._container.omega[self._index])

    @property
    def sx(self) -> np.ndarray:
        return self._container.sx_params[self._index]

    def focal(self) -> float:
        return float(np.mean(self.focal_length))

    def get_K(self) -> np.ndarray:
        fl = self.focal_length
        pp = self.principal_point
        return np.array([[fl[0], 0, pp[0]],
                         [0, fl[1], pp[1]],
                         [0, 0, 1]], dtype=np.float64)

    def fisheye_from_normal(self, uv: np.ndarray) -> np.ndarray:
        return _fisheye_from_normal(uv)

    def normal_from_fisheye(self, uv: np.ndarray) -> np.ndarray:
        return _normal_from_fisheye(uv)

    def Distortion(self, uv: np.ndarray) -> np.ndarray:
        return _camera_distortion(self.model_id, uv, self._k_values(), self.p, self.omega, self.sx)

    def img2cam(self, xy: np.ndarray) -> np.ndarray:
        return _camera_img2cam(self.model_id, xy, self.principal_point, self.focal_length,
                               self._k_values(), self.p, self.omega, self.sx)

    def cam2img(self, uvw: np.ndarray) -> np.ndarray:
        return _camera_cam2img(self.model_id, uvw, self.principal_point, self.focal_length,
                               self._k_values(), self.p, self.omega, self.sx)


class Cameras:
    """Struct-of-Arrays container for camera intrinsics."""

    def __init__(self, num_cameras: int = 0):
        self.num_cameras = num_cameras
        self.model_ids = np.full(num_cameras, CameraModelId.INVALID.value, dtype=np.int32)
        self.widths = np.zeros(num_cameras, dtype=np.int32)
        self.heights = np.zeros(num_cameras, dtype=np.int32)
        self.has_prior_focal_length = np.zeros(num_cameras, dtype=bool)
        self.focal_lengths = np.zeros((num_cameras, 2), dtype=np.float64)
        self.principal_points = np.zeros((num_cameras, 2), dtype=np.float64)
        self.k_params = np.zeros((num_cameras, 6), dtype=np.float64)
        self.k_counts = np.zeros(num_cameras, dtype=np.int32)
        self.p_params = np.zeros((num_cameras, 2), dtype=np.float64)
        self.omega = np.zeros(num_cameras, dtype=np.float64)
        self.sx_params = np.zeros((num_cameras, 2), dtype=np.float64)
        self.params = np.zeros((num_cameras, 16), dtype=np.float64)
        self.param_sizes = np.zeros(num_cameras, dtype=np.int32)

    def __len__(self) -> int:
        return self.num_cameras

    def __getitem__(self, index: int) -> CameraView:
        if index < 0 or index >= self.num_cameras:
            raise IndexError(f"Camera index {index} out of range [0, {self.num_cameras})")
        return CameraView(self, index)

    def __iter__(self):
        for idx in range(self.num_cameras):
            yield CameraView(self, idx)

    def values(self):
        return [CameraView(self, idx) for idx in range(self.num_cameras)]

    def _get_k_values(self, index: int) -> np.ndarray:
        count = self.k_counts[index]
        if count == 0:
            return np.zeros(1, dtype=np.float64)
        return self.k_params[index, :count]

    def _get_param_tuple(self, index: int):
        return (self.principal_points[index],
                self.focal_lengths[index],
                self._get_k_values(index),
                self.p_params[index],
                float(self.omega[index]),
                self.sx_params[index])

    def set_params(self, index: int, params: Union[np.ndarray, List[float]], model_id: Optional[CameraModelId] = None):
        params = np.asarray(params, dtype=np.float64)
        if model_id is not None:
            self.model_ids[index] = model_id.value if isinstance(model_id, CameraModelId) else int(model_id)
        info = get_camera_model_info(CameraModelId(int(self.model_ids[index])))
        count = len(params)
        self.param_sizes[index] = count
        self.params[index] = 0
        self.params[index, :count] = params

        # focal lengths
        focal_indices = info['focal']
        if len(focal_indices) == 1:
            f = params[focal_indices[0]]
            self.focal_lengths[index] = np.array([f, f], dtype=np.float64)
        elif len(focal_indices) == 2:
            self.focal_lengths[index, 0] = params[focal_indices[0]]
            self.focal_lengths[index, 1] = params[focal_indices[1]]

        # principal point
        pp_indices = info['pp']
        if len(pp_indices) == 2:
            self.principal_points[index, 0] = params[pp_indices[0]]
            self.principal_points[index, 1] = params[pp_indices[1]]

        # distortion coefficients
        k_indices = info['k']
        self.k_counts[index] = len(k_indices)
        self.k_params[index] = 0
        if len(k_indices) > 0:
            self.k_params[index, :len(k_indices)] = params[k_indices]

        p_indices = info['p']
        self.p_params[index] = 0
        if len(p_indices) == 2:
            self.p_params[index, 0] = params[p_indices[0]]
            self.p_params[index, 1] = params[p_indices[1]]

        omega_indices = info['omega']
        if len(omega_indices) == 1:
            self.omega[index] = params[omega_indices[0]]
        else:
            self.omega[index] = 0

        sx_indices = info['sx']
        self.sx_params[index] = 0
        if len(sx_indices) == 2:
            self.sx_params[index, 0] = params[sx_indices[0]]
            self.sx_params[index, 1] = params[sx_indices[1]]

    def img2cam(self, xy: np.ndarray, camera_indices: Union[int, np.ndarray]) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        input_was_single = xy.ndim == 1
        if input_was_single:
            xy = xy.reshape(1, 2)

        indices = np.asarray(camera_indices, dtype=np.int64)
        if indices.ndim == 0:
            params = self._get_param_tuple(int(indices))
            result = _camera_img2cam(CameraModelId(int(self.model_ids[int(indices)])), xy, *params)
            return result[0] if input_was_single else result
        indices = indices.reshape(-1)
        if indices.size == 1:
            params = self._get_param_tuple(int(indices[0]))
            result = _camera_img2cam(CameraModelId(int(self.model_ids[int(indices[0])])), xy, *params)
            return result[0] if input_was_single else result

        if indices.shape[0] != xy.shape[0]:
            raise ValueError("camera_indices must match number of points")

        result = np.zeros_like(xy)
        unique_indices = np.unique(indices)
        for cam_idx in unique_indices:
            mask = indices == cam_idx
            params = self._get_param_tuple(int(cam_idx))
            result[mask] = _camera_img2cam(CameraModelId(int(self.model_ids[int(cam_idx)])), xy[mask], *params)
        return result[0] if input_was_single else result

    def cam2img(self, uvw: np.ndarray, camera_indices: Union[int, np.ndarray]) -> np.ndarray:
        uvw = np.asarray(uvw, dtype=np.float64)
        input_was_single = uvw.ndim == 1
        if input_was_single:
            uvw = uvw.reshape(1, 3)

        indices = np.asarray(camera_indices, dtype=np.int64)
        if indices.ndim == 0:
            params = self._get_param_tuple(int(indices))
            result = _camera_cam2img(CameraModelId(int(self.model_ids[int(indices)])), uvw, *params)
            return result[0] if input_was_single else result
        indices = indices.reshape(-1)
        if indices.size == 1:
            params = self._get_param_tuple(int(indices[0]))
            result = _camera_cam2img(CameraModelId(int(self.model_ids[int(indices[0])])), uvw, *params)
            return result[0] if input_was_single else result

        if indices.shape[0] != uvw.shape[0]:
            raise ValueError("camera_indices must match number of points")

        result = np.zeros((uvw.shape[0], 2), dtype=np.float64)
        unique_indices = np.unique(indices)
        for cam_idx in unique_indices:
            mask = indices == cam_idx
            params = self._get_param_tuple(int(cam_idx))
            result[mask] = _camera_cam2img(CameraModelId(int(self.model_ids[int(cam_idx)])), uvw[mask], *params)
        return result[0] if input_was_single else result

def _fisheye_from_normal(uv: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(uv, axis=-1, keepdims=True)
    r = np.clip(r, 1e-8, None)
    theta = np.arctan(r)
    return uv * theta / r


def _normal_from_fisheye(uv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(uv, axis=-1, keepdims=True)
    theta_cos_theta = theta * np.cos(theta)
    theta_cos_theta = np.clip(theta_cos_theta, 1e-8, None)
    return uv * np.sin(theta) / theta_cos_theta


def _camera_distortion(model_id: CameraModelId,
                       uv: np.ndarray,
                       k: np.ndarray,
                       p: np.ndarray,
                       omega: float,
                       sx: np.ndarray) -> np.ndarray:
    if model_id == CameraModelId.SIMPLE_RADIAL:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        return uv * k[0] * r2
    elif model_id == CameraModelId.RADIAL:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        return uv * k[0] * r2 + uv * k[1] * r2 ** 2
    elif model_id == CameraModelId.OPENCV:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        uv_prod = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
        radial = k[0] * r2 + k[1] * r2 ** 2
        d = uv * radial + 2 * p * uv_prod
        d += p[::-1] * (r2 + 2 * uv ** 2)
        return d
    elif model_id == CameraModelId.OPENCV_FISHEYE:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        radial = k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
        return uv * radial
    elif model_id == CameraModelId.FULL_OPENCV:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        uv_prod = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
        numerator = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
        denominator = 1 + k[3] * r2 + k[4] * r2 ** 2 + k[5] * r2 ** 3
        radial = numerator / np.clip(denominator, 1e-8, None) - 1
        d = uv * radial + 2 * p * uv_prod
        d += p[::-1] * (r2 + 2 * uv ** 2)
        return d
    elif model_id == CameraModelId.FOV:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        omega2 = omega ** 2
        epsilon = 1e-4
        factor = np.zeros_like(r2)
        if omega2 < epsilon:
            factor = (omega2 * r2) / 3 - omega2 / 12 + 1
        else:
            r2_mask = r2 < epsilon
            tan_half_omega = np.tan(omega / 2)
            factor[r2_mask] = (
                -2 * tan_half_omega * (4 * r2[r2_mask] * tan_half_omega ** 2 - 3)
            ) / (3 * omega)
            r2_mask_inv = ~r2_mask
            radius = np.sqrt(r2[r2_mask_inv])
            numerator = np.arctan(radius * 2 * tan_half_omega)
            factor[r2_mask_inv] = numerator / (radius * omega)
        return uv * factor
    elif model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        return uv * k[0] * r2
    elif model_id == CameraModelId.RADIAL_FISHEYE:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        return uv * k[0] * r2 + uv * k[1] * r2 ** 2
    elif model_id == CameraModelId.THIN_PRISM_FISHEYE:
        r2 = np.sum(uv ** 2, axis=-1, keepdims=True)
        uv_prod = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
        radial = k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
        d = uv * radial + 2 * p * uv_prod
        d += p[::-1] * (r2 + 2 * uv ** 2)
        d += sx * r2
        return d
    else:
        raise NotImplementedError


def _camera_img2cam(model_id: CameraModelId,
                    xy: np.ndarray,
                    principal_point: np.ndarray,
                    focal_length: np.ndarray,
                    k: np.ndarray,
                    p: np.ndarray,
                    omega: float,
                    sx: np.ndarray) -> np.ndarray:
    if model_id == CameraModelId.SIMPLE_PINHOLE:
        f = float(np.mean(focal_length))
        return (xy - principal_point) / f
    elif model_id == CameraModelId.PINHOLE:
        return (xy - principal_point) / focal_length
    elif model_id == CameraModelId.FOV:
        uv = (xy - principal_point) / focal_length
        return _camera_distortion(model_id, uv, k, p, omega, sx)
    elif model_id in (CameraModelId.SIMPLE_RADIAL, CameraModelId.RADIAL,
                      CameraModelId.OPENCV, CameraModelId.OPENCV_FISHEYE,
                      CameraModelId.FULL_OPENCV, CameraModelId.SIMPLE_RADIAL_FISHEYE,
                      CameraModelId.RADIAL_FISHEYE, CameraModelId.THIN_PRISM_FISHEYE):
        K = np.array([[focal_length[0], 0, principal_point[0]],
                      [0, focal_length[1], principal_point[1]],
                      [0, 0, 1]], dtype=np.float64)
        xy_expanded = np.expand_dims(xy, axis=1)
        if model_id in (CameraModelId.SIMPLE_RADIAL, CameraModelId.SIMPLE_RADIAL_FISHEYE):
            dist_coeffs = np.array([k[0], 0, 0, 0], dtype=np.float64)
            uv = cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
            if model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
                return _normal_from_fisheye(uv)
            return uv
        elif model_id in (CameraModelId.RADIAL, CameraModelId.RADIAL_FISHEYE):
            dist_coeffs = np.array([k[0], k[1], 0, 0], dtype=np.float64)
            uv = cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
            if model_id == CameraModelId.RADIAL_FISHEYE:
                return _normal_from_fisheye(uv)
            return uv
        elif model_id == CameraModelId.OPENCV:
            dist_coeffs = np.array([k[0], k[1], p[0], p[1]], dtype=np.float64)
            return cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
        elif model_id == CameraModelId.OPENCV_FISHEYE:
            dist_coeffs = np.array([k[0], k[1], 0, 0, k[2]], dtype=np.float64)
            uv = cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
            return _normal_from_fisheye(uv)
        elif model_id == CameraModelId.FULL_OPENCV:
            dist_coeffs = np.array([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5]], dtype=np.float64)
            return cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
        elif model_id == CameraModelId.THIN_PRISM_FISHEYE:
            dist_coeffs = np.array([k[0], k[1], p[0], p[1], k[2], 0, 0, 0, sx[0], sx[1], 0, 0], dtype=np.float64)
            uv = cv2.undistortPoints(xy_expanded, K, dist_coeffs).reshape(-1, 2)
            return _normal_from_fisheye(uv)
    else:
        raise NotImplementedError


def _camera_cam2img(model_id: CameraModelId,
                    uvw: np.ndarray,
                    principal_point: np.ndarray,
                    focal_length: np.ndarray,
                    k: np.ndarray,
                    p: np.ndarray,
                    omega: float,
                    sx: np.ndarray) -> np.ndarray:
    uv = uvw[..., :2] / (np.expand_dims(uvw[..., 2], axis=-1) + 1e-10)
    f_scalar = float(np.mean(focal_length))
    if model_id == CameraModelId.SIMPLE_PINHOLE:
        return uv * f_scalar + principal_point
    elif model_id == CameraModelId.PINHOLE:
        return uv * focal_length + principal_point
    elif model_id in (CameraModelId.SIMPLE_RADIAL, CameraModelId.RADIAL,
                      CameraModelId.OPENCV, CameraModelId.FULL_OPENCV):
        uv += _camera_distortion(model_id, uv, k, p, omega, sx)
        scale = f_scalar if model_id in (CameraModelId.SIMPLE_RADIAL, CameraModelId.RADIAL) else focal_length
        return uv * scale + principal_point
    elif model_id == CameraModelId.OPENCV_FISHEYE:
        uv = _fisheye_from_normal(uv)
        uv += _camera_distortion(model_id, uv, k, p, omega, sx)
        return uv * focal_length + principal_point
    elif model_id == CameraModelId.FOV:
        uv = _camera_distortion(model_id, uv, k, p, omega, sx)
        return uv * f_scalar + principal_point
    elif model_id in (CameraModelId.SIMPLE_RADIAL_FISHEYE, CameraModelId.RADIAL_FISHEYE, CameraModelId.THIN_PRISM_FISHEYE):
        uv = _fisheye_from_normal(uv)
        uv += _camera_distortion(model_id, uv, k, p, omega, sx)
        scale = f_scalar if model_id != CameraModelId.THIN_PRISM_FISHEYE else focal_length
        return uv * scale + principal_point
    else:
        raise NotImplementedError


# ============================================================================
# View Graph
# ============================================================================

class ViewGraph:
    def __init__(self):
        self.image_pairs = {} # includes: (image_id1, image_id2) -> ImagePair
        self.num_images = 0
        self.num_pairs = 0

    def establish_adjacency_list(self):
        self.adjacency_list = {}
        for pair in self.image_pairs.values():
            if not pair.is_valid:
                continue
            if pair.image_id1 not in self.adjacency_list:
                self.adjacency_list[pair.image_id1] = set()
            self.adjacency_list[pair.image_id1].add(pair.image_id2)
            if pair.image_id2 not in self.adjacency_list:
                self.adjacency_list[pair.image_id2] = set()
            self.adjacency_list[pair.image_id2].add(pair.image_id1)

    def BFS(self, root):
        q = Queue()
        q.put(root)
        self.visited[root] = True
        component = [root]

        while not q.empty():
            current = q.get()
            for neighbor in self.adjacency_list[current]:
                if not self.visited[neighbor]:
                    q.put(neighbor)
                    self.visited[neighbor] = True
                    component.append(neighbor)

        return component

    def find_connected_component(self):
        self.connected_component = []
        self.visited = {}
        for image_id in self.adjacency_list.keys():
            self.visited[image_id] = False

        for image_id in self.adjacency_list.keys():
            if not self.visited[image_id]:
                component = self.BFS(image_id)
                self.connected_component.append(component)

    def keep_largest_connected_component(self, images: Images) -> bool:
        """Keep only the largest connected component.
        
        Each image is evaluated individually. For multi-camera rigs,
        rig-level filtering should be applied at export time.
        
        Args:
            images: Either list of Image objects or Images container.
            
        Returns:
            True if successful, False otherwise.
        """
        self.establish_adjacency_list()
        self.find_connected_component()

        max_idx = -1
        max_img = 0
        for idx, component in enumerate(self.connected_component):
            if len(component) > max_img:
                max_img = len(component)
                max_idx = idx

        if max_idx == -1:
            return False

        largest_component = self.connected_component[max_idx]
        
        # Always use individual camera registration
        # Multi-camera rig filtering is handled at export time
        for i in range(len(images)):
            images.is_registered[i] = i in largest_component

        for pair in self.image_pairs.values():
            img1_reg = images[pair.image_id1].is_registered
            img2_reg = images[pair.image_id2].is_registered
            
            if not img1_reg or not img2_reg:
                pair.is_valid = False
        return True

    def mark_connected_components(self, images: Images) -> int:
        """Mark each image with its connected component ID.
        
        Args:
            images: Either list of Image objects or Images container.
            
        Returns:
            Number of connected components.
        """
        self.establish_adjacency_list()
        self.find_connected_component()

        cluster_num_img = []
        for comp in range(len(self.connected_component)):
            cluster_num_img.append((len(self.connected_component[comp]), comp))
        cluster_num_img.sort(key=lambda x: x[0], reverse=True)

        for i in range(len(images)):
            images[i].cluster_id = -1
        
        for comp in range(len(cluster_num_img)):
            for image_id in self.connected_component[cluster_num_img[comp][1]]:
                images[image_id].cluster_id = comp
        
        return comp + 1


# ============================================================================
# Semantic Object Management
# ============================================================================

class Object:
    """Represents a physical object tracked across multiple images.
    
    Each object has a global ID and is associated with different semantic 
    label IDs in different images (since segmentation is done per-camera).
    """
    def __init__(self, object_id: int):
        self.object_id = object_id  # Global unique object ID
        
        # Map from image_id to local semantic label_id in that image
        # {image_id: label_id}
        self.image_labels: Dict[int, int] = {}
        
        # Statistics for dynamic object detection
        self.observation_count = 0  # Number of times seen in image pairs
        self.inlier_ratios: List[float] = []  # RANSAC inlier ratios
        self.is_dynamic = False  # Whether marked as dynamic
        
        # Associated features and tracks
        self.feature_observations: List[Tuple[int, int]] = []  # [(image_id, feat_idx), ...]
        self.track_ids: List[int] = []  # IDs of tracks on this object
    
    def add_observation(self, image_id: int, label_id: int):
        """Associate this object with a semantic label in an image."""
        self.image_labels[image_id] = label_id
    
    def get_label_in_image(self, image_id: int) -> Optional[int]:
        """Get the semantic label ID for this object in a specific image."""
        return self.image_labels.get(image_id, None)
    
    def has_observation_in_image(self, image_id: int) -> bool:
        """Check if this object is visible in an image."""
        return image_id in self.image_labels
    
    def add_inlier_ratio(self, ratio: float):
        """Add an inlier ratio observation for dynamic detection."""
        self.inlier_ratios.append(ratio)
        self.observation_count += 1
    
    def get_inlier_statistics(self) -> Dict:
        """Get statistics of inlier ratios."""
        if len(self.inlier_ratios) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'p25': 0.0,
                'p75': 0.0
            }
        
        ratios = np.array(self.inlier_ratios)
        return {
            'count': len(ratios),
            'mean': float(np.mean(ratios)),
            'median': float(np.median(ratios)),
            'p25': float(np.percentile(ratios, 25)),
            'p75': float(np.percentile(ratios, 75))
        }


class Objects:
    """Container for managing all detected objects across images.
    
    Handles object tracking across different camera views where semantic
    segmentation is performed independently per camera.
    """
    def __init__(self):
        self.objects: Dict[int, Object] = {}  # object_id -> Object
        self.next_object_id = 1
        
        # Reverse mapping: (image_id, label_id) -> object_id
        self.label_to_object: Dict[Tuple[int, int], int] = {}
        
        # Static label (background)
        self.background_label = 0
    
    def create_object(self) -> Object:
        """Create a new object with a unique ID."""
        obj = Object(self.next_object_id)
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj
    
    def add_observation(self, object_id: int, image_id: int, label_id: int):
        """Associate an object with a semantic label in an image."""
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} does not exist")
        
        self.objects[object_id].add_observation(image_id, label_id)
        self.label_to_object[(image_id, label_id)] = object_id
    
    def get_object_id(self, image_id: int, label_id: int) -> Optional[int]:
        """Get the global object ID for a semantic label in an image."""
        return self.label_to_object.get((image_id, label_id), None)
    
    def get_object(self, object_id: int) -> Optional[Object]:
        """Get an object by its ID."""
        return self.objects.get(object_id, None)
    
    def get_objects_in_image(self, image_id: int) -> List[int]:
        """Get all object IDs visible in an image."""
        return [
            obj_id for obj_id, obj in self.objects.items()
            if obj.has_observation_in_image(image_id)
        ]
    
    def get_dynamic_objects(self) -> List[int]:
        """Get IDs of all objects marked as dynamic."""
        return [obj_id for obj_id, obj in self.objects.items() if obj.is_dynamic]
    
    def mark_dynamic(self, object_id: int):
        """Mark an object as dynamic."""
        if object_id in self.objects:
            self.objects[object_id].is_dynamic = True
    
    def __len__(self) -> int:
        """Number of tracked objects."""
        return len(self.objects)
    
    def __contains__(self, object_id: int) -> bool:
        """Check if an object exists."""
        return object_id in self.objects
