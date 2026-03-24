"""
Optimization models for bundle adjustment and global positioning.
This module contains PyTorch model classes used in LM optimization,
following the pattern in bae/ba_helpers.py.
"""

import torch
from torch import nn
import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform


@map_transform
def calc_pose(ref_poses, rel_poses):
    """Calculate final camera pose from reference and relative poses."""
    ref_poses_se3 = pp.SE3(ref_poses)
    rel_poses_se3 = pp.SE3(rel_poses)
    return rel_poses_se3 @ ref_poses_se3


@map_transform
def calc_trans(ref_R, ref_t, rel_R, rel_t):
    """Calculate final camera pose from separate reference and relative rotations and translations."""
    ref_rots_so3 = pp.SO3(ref_R)
    rel_rots_so3 = pp.SO3(rel_R)
    ref_rots_inv = ref_rots_so3.Inv()
    rel_rots_inv = rel_rots_so3.Inv()
    pose_trans = -(ref_rots_inv @ (rel_rots_inv @ rel_t + ref_t))
    return pose_trans


def _set_intrinsics_param(module, attr_name, camera_intrs, optimize_intrinsics):
    intrinsics = TrackingTensor(camera_intrs)
    if optimize_intrinsics:
        module.register_parameter(attr_name, nn.Parameter(intrinsics))
    else:
        module.register_buffer(attr_name, intrinsics)


class ReprojectionModel(nn.Module):
    """
    Single-camera bundle adjustment model.
    Optimizes camera extrinsics, intrinsics, and 3D points to minimize reprojection error.
    
    Similar to bae.ba_helpers.Reproj but adapted for instantsfm's camera models.
    """
    def __init__(self, image_extrs, camera_intrs, points_3d, cost_fn, optimize_intrinsics=True):
        """
        Args:
            image_extrs: [num_imgs, 7] SE3 poses as quaternion + translation
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
        """
        super().__init__()
        self.extrinsics = nn.Parameter(TrackingTensor(image_extrs))
        _set_intrinsics_param(self, "intrinsics", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.extrinsics.trim_SE3_grad = True
        self.cost_fn = cost_fn

    def forward(self, points_2d, image_indices, camera_indices, point_indices, camera_pps):
        """
        Forward pass computing reprojection residuals (without depth constraint).
        
        Args:
            points_2d: [N, 2] observed 2D points
            image_indices: [N] image index for each observation
            camera_indices: [N] camera index for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            
        Returns:
            [N, 2] reprojection residuals
        """
        # cost_fn returns [N, 2] with (x, y) only
        loss = self.cost_fn(
            self.points_3d[point_indices], 
            self.extrinsics[image_indices],
            self.intrinsics[camera_indices], 
            camera_pps[camera_indices]
        )
        loss = loss - points_2d
        return loss


class ReprojectionModelWithDepth(nn.Module):
    """
    Single-camera bundle adjustment model with depth constraint.
    Optimizes camera extrinsics, intrinsics, and 3D points to minimize 
    reprojection error and depth error.
    """
    def __init__(
        self,
        image_extrs,
        camera_intrs,
        points_3d,
        cost_fn,
        depth_weight,
        optimize_intrinsics=True,
    ):
        """
        Args:
            image_extrs: [num_imgs, 7] SE3 poses as quaternion + translation
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
            depth_weight: weight for depth constraint relative to reprojection error
        """
        super().__init__()
        self.extrinsics = nn.Parameter(TrackingTensor(image_extrs))
        _set_intrinsics_param(self, "intrinsics", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.extrinsics.trim_SE3_grad = True
        self.cost_fn = cost_fn
        self.depth_weight = depth_weight

    def forward(self, points_2d, image_indices, camera_indices, point_indices, camera_pps, depths_ref):
        """
        Forward pass computing reprojection and depth residuals.
        
        Args:
            points_2d: [N, 2] observed 2D points
            image_indices: [N] image index for each observation
            camera_indices: [N] camera index for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            depths_ref: [N] reference inverse depth values (already 1/depth)
            
        Returns:
            [N, 3] reprojection and depth residuals (x_error, y_error, inv_depth_error)
        """
        # cost_fn returns [N, 3] with (x, y, depth)
        loss = self.cost_fn(
            self.points_3d[point_indices], 
            self.extrinsics[image_indices],
            self.intrinsics[camera_indices], 
            camera_pps[camera_indices]
        )
        
        # Compute reprojection error in-place
        loss[..., :2] = loss[..., :2] - points_2d
        
        # Compute inverse depth error with weight
        # depths_ref is already inverse depth (1/depth)
        eps = 1e-6  # Small epsilon to avoid division by zero
        inv_depth_proj = 1.0 / (loss[..., 2] + eps)
        loss[..., 2] = (inv_depth_proj - depths_ref) * self.depth_weight
        
        return loss


class ReprojectionMultiRigModel(nn.Module):
    """
    Multi-camera rig bundle adjustment model.
    Uses reference + relative poses for camera rigs.
    """
    def __init__(self, camera_intrs, points_3d, ref_poses, rel_poses, cost_fn, optimize_intrinsics=True):
        """
        Args:
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            ref_poses: [num_groups, 7] reference poses for each rig group
            rel_poses: [num_positions, 7] relative poses for each camera position
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
            optimize_rel_poses: whether to optimize relative poses (False for fixed rig)
        """
        super().__init__()
        _set_intrinsics_param(self, "intrs", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_poses = nn.Parameter(TrackingTensor(ref_poses))
        self.rel_poses = nn.Parameter(TrackingTensor(rel_poses))
        self.ref_poses.trim_SE3_grad = True
        self.rel_poses.trim_SE3_grad = True
        self.cost_fn = cost_fn

    def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps):
        """
        Forward pass computing reprojection residuals for multi-rig setup (without depth constraint).
        
        Args:
            points_2d: [N, 2] observed 2D points
            camera_indices: [N] camera index for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            
        Returns:
            [N, 2] reprojection residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        image_poses = calc_pose(self.ref_poses[group_idx], self.rel_poses[member_idx])
        
        # cost_fn returns [N, 2] with (x, y) only
        loss = self.cost_fn(
            self.points_3d[point_indices],
            image_poses,
            self.intrs[camera_indices],
            camera_pps[camera_indices]
        )
        loss = loss - points_2d
        return loss


class ReprojectionMultiRigModelWithDepth(nn.Module):
    """
    Multi-camera rig bundle adjustment model with depth constraint.
    Uses reference + relative poses for camera rigs.
    """
    def __init__(
        self,
        camera_intrs,
        points_3d,
        ref_poses,
        rel_poses,
        cost_fn,
        depth_weight,
        optimize_intrinsics=True,
    ):
        """
        Args:
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            ref_poses: [num_groups, 7] reference poses for each rig group
            rel_poses: [num_positions, 7] relative poses for each camera position
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
            depth_weight: weight for depth constraint relative to reprojection error
            optimize_rel_poses: whether to optimize relative poses (False for fixed rig)
        """
        super().__init__()
        _set_intrinsics_param(self, "intrs", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_poses = nn.Parameter(TrackingTensor(pp.SE3(ref_poses)))
        self.rel_poses = nn.Parameter(TrackingTensor(pp.SE3(rel_poses)))
        self.ref_poses.trim_SE3_grad = True
        self.rel_poses.trim_SE3_grad = True
        self.cost_fn = cost_fn
        self.depth_weight = depth_weight

    def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps, depths_ref):
        """
        Forward pass computing reprojection and depth residuals for multi-rig setup.
        
        Args:
            points_2d: [N, 2] observed 2D points
            camera_indices: [N] camera index for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            depths_ref: [N] reference inverse depth values (already 1/depth)
            
        Returns:
            [N, 3] reprojection and depth residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        image_poses = calc_pose(self.ref_poses[group_idx], self.rel_poses[member_idx])
        
        # cost_fn returns [N, 3] with (x, y, depth)
        loss = self.cost_fn(
            self.points_3d[point_indices],
            image_poses,
            self.intrs[camera_indices],
            camera_pps[camera_indices]
        )
        
        # Compute reprojection error in-place
        loss[..., :2] = loss[..., :2] - points_2d
        
        # Compute inverse depth error with weight
        # depths_ref is already inverse depth (1/depth)
        eps = 1e-6  # Small epsilon to avoid division by zero
        inv_depth_proj = 1.0 / (loss[..., 2] + eps)
        loss[..., 2] = (inv_depth_proj - depths_ref) * self.depth_weight
        
        return loss


class ReprojectionMultiRigModelFixedRel(nn.Module):
    """
    Multi-camera rig bundle adjustment model with FIXED relative poses.
    Only optimizes reference poses, not relative poses (for known rig geometry).
    """
    def __init__(self, camera_intrs, points_3d, ref_poses, cost_fn, optimize_intrinsics=True):
        """
        Args:
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            ref_poses: [num_groups, 7] reference poses for each rig group
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
        Note: rel_poses are passed via forward() as fixed input
        """
        super().__init__()
        _set_intrinsics_param(self, "intrs", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_poses = nn.Parameter(TrackingTensor(ref_poses))
        self.ref_poses.trim_SE3_grad = True
        self.cost_fn = cost_fn

    def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps, rel_poses):
        """
        Forward pass computing reprojection residuals for multi-rig setup (without depth constraint).
        
        Args:
            points_2d: [N, 2] observed 2D points
            camera_indices: [N] camera index for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            rel_poses: [num_positions, 7] FIXED relative poses for each camera position
            
        Returns:
            [N, 2] reprojection residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        image_poses = calc_pose(self.ref_poses[group_idx], rel_poses[member_idx])
        
        # cost_fn returns [N, 2] with (x, y) only
        loss = self.cost_fn(
            self.points_3d[point_indices],
            image_poses,
            self.intrs[camera_indices],
            camera_pps[camera_indices]
        )
        loss = loss - points_2d
        return loss


class ReprojectionMultiRigModelWithDepthFixedRel(nn.Module):
    """
    Multi-camera rig bundle adjustment model with depth constraint and FIXED relative poses.
    Only optimizes reference poses, not relative poses (for known rig geometry).
    """
    def __init__(
        self,
        camera_intrs,
        points_3d,
        ref_poses,
        cost_fn,
        depth_weight,
        optimize_intrinsics=True,
    ):
        """
        Args:
            camera_intrs: [num_cams, x] intrinsic parameters (excluding principal point)
            points_3d: [num_pts, 3] 3D point positions
            ref_poses: [num_groups, 7] reference poses for each rig group
            cost_fn: reprojection function from cost_function.py
            optimize_poses: whether to optimize camera poses
            depth_weight: weight for depth constraint relative to reprojection error
        Note: rel_poses are passed via forward() as fixed input
        """
        super().__init__()
        _set_intrinsics_param(self, "intrs", camera_intrs, optimize_intrinsics)
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_poses = nn.Parameter(TrackingTensor(pp.SE3(ref_poses)))
        self.ref_poses.trim_SE3_grad = True
        self.cost_fn = cost_fn
        self.depth_weight = depth_weight

    def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps, rel_poses, depths_ref):
        """
        Forward pass computing reprojection and depth residuals for multi-rig setup.
        
        Args:
            points_2d: [N, 2] observed 2D points
            camera_indices: [N] camera index for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            camera_pps: [num_cams, 2] principal points
            rel_poses: [num_positions, 7] FIXED relative poses for each camera position
            depths_ref: [N] reference inverse depth values (already 1/depth)
            
        Returns:
            [N, 3] reprojection and depth residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        image_poses = calc_pose(self.ref_poses[group_idx], rel_poses[member_idx])
        
        # cost_fn returns [N, 3] with (x, y, depth)
        loss = self.cost_fn(
            self.points_3d[point_indices],
            image_poses,
            self.intrs[camera_indices],
            camera_pps[camera_indices]
        )
        
        # Compute reprojection error in-place
        loss[..., :2] = loss[..., :2] - points_2d
        
        # Compute inverse depth error with weight
        # depths_ref is already inverse depth (1/depth)
        eps = 1e-6  # Small epsilon to avoid division by zero
        inv_depth_proj = 1.0 / (loss[..., 2] + eps)
        loss[..., 2] = (inv_depth_proj - depths_ref) * self.depth_weight
        
        return loss


class PairwiseSingleCameraModel(nn.Module):
    """
    Global positioning model for single-camera sequences.
    Uses pairwise depth constraints to position cameras and 3D points.
    
    Similar to pose graph optimization but with scale parameters.
    """
    def __init__(self, image_translations, points_3d, scales, cost_fn, scale_indices=None):
        """
        Args:
            image_translations: [num_imgs, 3] camera translation vectors
            points_3d: [num_pts, 3] 3D point positions
            scales: [N, 1] depth scales for each observation
            cost_fn: pairwise cost function from cost_function.py
            scale_indices: indices of scales that should NOT be optimized (fixed depth)
        """
        super().__init__()
        self.translations = nn.Parameter(TrackingTensor(image_translations))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.scales = nn.Parameter(TrackingTensor(scales))
        self.cost_fn = cost_fn
        
        if scale_indices is not None:
            all_indices = torch.arange(scales.shape[0], device=scales.device)
            self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

    def forward(self, translations, image_indices, point_indices, is_calibrated):
        """
        Forward pass computing pairwise positioning residuals.
        
        Args:
            translations: [N, 3] ray directions for each observation
            image_indices: [N] image index for each observation
            point_indices: [N] 3D point index for each observation
            is_calibrated: [num_imgs] whether each camera is calibrated
            
        Returns:
            [N, 3] pairwise positioning residuals
        """
        loss = self.cost_fn(
            self.points_3d[point_indices],
            self.translations[image_indices],
            self.scales,
            translations,
            is_calibrated[image_indices]
        )
        return loss


class PairwiseMultiRigModel(nn.Module):
    """
    Global positioning model for multi-camera rigs.
    Uses reference + relative translations for camera rigs with full pose computation.
    
    Structure similar to ReprojectionMultiRigModel but for positioning stage.
    """
    def __init__(self, points_3d, scales, ref_trans, rel_trans, cost_fn, scale_indices=None):
        """
        Args:
            points_3d: [num_pts, 3] 3D point positions
            scales: [N, 1] depth scales for each observation
            ref_trans: [num_groups, 3] reference translations for each rig group (optimizable)
            rel_trans: [num_positions, 3] relative translations for each camera position (optimizable)
            cost_fn: pairwise cost function from cost_function.py
            scale_indices: indices of scales that should NOT be optimized
        """
        super().__init__()
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_trans = nn.Parameter(TrackingTensor(ref_trans))
        self.rel_trans = nn.Parameter(TrackingTensor(rel_trans))
        self.scales = nn.Parameter(TrackingTensor(scales))
        self.cost_fn = cost_fn
        
        if scale_indices is not None:
            all_indices = torch.arange(scales.shape[0], device=scales.device)
            self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

    def forward(self, feature_undist, grouping_indices, point_indices, is_calibrated, ref_rots, rel_rots):
        """
        Forward pass computing pairwise positioning residuals for multi-rig.
        
        Args:
            feature_undist: [N, 3] undistorted feature directions for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            is_calibrated: [num_groups] whether each rig group is calibrated
            ref_rots: [num_groups, 4] reference rotations for each rig group (fixed)
            rel_rots: [num_positions, 4] relative rotations for each camera position (fixed)
            
        Returns:
            [N, 3] pairwise positioning residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        
        # Compute full poses: world2rel = ref2rel @ world2ref
        ref_R = ref_rots[group_idx]  # [N, 4]
        rel_R = rel_rots[member_idx]  # [N, 4]
        ref_t = self.ref_trans[group_idx]  # [N, 3]
        rel_t = self.rel_trans[member_idx]  # [N, 3]

        # actually we solve cam2world representation
        pose_R = pp.SO3(rel_R) @ pp.SO3(ref_R)
        pose_t = calc_trans(ref_R, ref_t, rel_R, rel_t)
        
        translations = pose_R.Inv() @ feature_undist
        calib_mask = is_calibrated[group_idx]
        loss = self.cost_fn(
            self.points_3d[point_indices],
            pose_t,
            self.scales,
            translations,
            calib_mask
        )
        return loss


class PairwiseMultiRigModelFixedRel(nn.Module):
    """
    Global positioning model for multi-camera rigs with FIXED relative translations.
    Only optimizes reference translations, not relative translations.
    Uses full pose computation like PairwiseMultiRigModel.
    
    Structure similar to PairwiseMultiRigModel but rel_trans is passed as fixed input.
    """
    def __init__(self, points_3d, scales, ref_trans, cost_fn, scale_indices=None):
        """
        Args:
            points_3d: [num_pts, 3] 3D point positions
            scales: [N, 1] depth scales for each observation
            ref_trans: [num_groups, 3] reference translations for each rig group (optimizable)
            cost_fn: pairwise cost function from cost_function.py
            scale_indices: indices of scales that should NOT be optimized
        Note: rel_trans, ref_rots, rel_rots are passed via forward() as fixed inputs
        """
        super().__init__()
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.ref_trans = nn.Parameter(TrackingTensor(ref_trans))
        self.scales = nn.Parameter(TrackingTensor(scales))
        self.cost_fn = cost_fn
        
        if scale_indices is not None:
            all_indices = torch.arange(scales.shape[0], device=scales.device)
            self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

    def forward(self, feature_undist, grouping_indices, point_indices, is_calibrated, ref_rots, rel_rots, rel_trans):
        """
        Forward pass computing pairwise positioning residuals for multi-rig.
        
        Args:
            feature_undist: [N, 3] undistorted feature directions for each observation
            grouping_indices: [N, 2] (group_idx, member_idx) for each observation
            point_indices: [N] 3D point index for each observation
            is_calibrated: [num_groups] whether each rig group is calibrated
            ref_rots: [num_groups, 4] reference rotations for each rig group (fixed)
            rel_rots: [num_positions, 4] relative rotations for each camera position (fixed)
            rel_trans: [num_positions, 3] FIXED relative translations for each camera position
            
        Returns:
            [N, 3] pairwise positioning residuals
        """
        group_idx = grouping_indices[:, 0]
        member_idx = grouping_indices[:, 1]
        
        # Compute full poses: world2rel = ref2rel @ world2ref
        ref_R = ref_rots[group_idx]  # [N, 4]
        rel_R = rel_rots[member_idx]  # [N, 4]
        ref_t = self.ref_trans[group_idx]  # [N, 3]
        rel_t = rel_trans[member_idx]  # [N, 3]

        # actually we solve cam2world representation
        pose_R = pp.SO3(rel_R) @ pp.SO3(ref_R)
        pose_t = calc_trans(ref_R, ref_t, rel_R, rel_t)
        
        translations = pose_R.Inv() @ feature_undist
        calib_mask = is_calibrated[group_idx]
        loss = self.cost_fn(
            self.points_3d[point_indices],
            pose_t,
            self.scales,
            translations,
            calib_mask
        )
        return loss


class FetzerCalibrationModel(nn.Module):
    """
    View graph calibration model using Fetzer's method.
    Optimizes focal lengths from relative poses.
    """
    def __init__(self, focals, cost_fn):
        """
        Args:
            focals: [num_cameras, 1] initial focal lengths
            cost_fn: fetzer cost function from cost_function.py
        """
        super().__init__()
        self.focals = nn.Parameter(TrackingTensor(focals))
        self.cost_fn = cost_fn

    def forward(self, ds, camera_indices1, camera_indices2):
        """
        Forward pass computing focal length calibration residuals.
        
        Args:
            ds: [num_pairs, 1, 3, 4] Fetzer's constraint matrices
            camera_indices1: [num_pairs] first camera index for each pair
            camera_indices2: [num_pairs] second camera index for each pair
            
        Returns:
            [num_pairs, 2] calibration residuals
        """
        loss = self.cost_fn(
            self.focals[camera_indices1],
            self.focals[camera_indices2],
            ds
        )
        return loss
