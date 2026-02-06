import numpy as np
import tqdm

from instantsfm.utils.cost_function import pairwise_cost
from instantsfm.scene.rig_utils import Rig2Single
from instantsfm.utils.optimization_models import PairwiseSingleCameraModel, PairwiseMultiRigModel, PairwiseMultiRigModelFixedRel

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Huber
from bae.utils.pysolvers import PCG
from bae.optim import LM
from bae.autograd.function import TrackingTensor


def _prepare_single_camera_data(cameras, images, tracks, device, min_observations, use_depths):
    """Prepare all data for single camera optimization."""
    # Filter tracks by observation count
    valid_mask = np.array([tracks.observations[i].shape[0] >= min_observations 
                           for i in range(len(tracks))])
    tracks.filter_by_mask(valid_mask)
    
    # Filter images that have no tracks
    image_used = np.zeros(len(images), dtype=bool)
    for track_id in range(len(tracks)):
        unique_image_ids = np.unique(tracks.observations[track_id][:, 0])
        image_used[unique_image_ids] = True
        if all(image_used):
            break
    images.is_registered[:] = images.is_registered & image_used
    
    # Build image ID mappings
    image_id2idx = {}
    image_idx2id = {}
    registered_indices = images.get_registered_indices()
    for idx, image_id in enumerate(registered_indices):
        image_id2idx[image_id] = idx
        image_idx2id[idx] = image_id
    
    # Prepare tensors
    image_translations = torch.tensor(images.world2cams[registered_indices, :3, 3], 
                                     dtype=torch.float64, device=device)
    points_3d = torch.tensor(tracks.xyzs, dtype=torch.float64, device=device)
    
    # Collect observation data
    translations_list = []
    image_indices_list = []
    point_indices_list = []
    depth_values_list = []
    depth_availability_list = []

    for track_id in range(len(tracks)):
        track_obs = tracks.observations[track_id]
        for image_id, feature_id in track_obs:
            if not images.is_registered[image_id]:
                continue
            if use_depths:
                depth = images.depths[image_id][feature_id]
                available = depth > 0
                depth_values_list.append(1.0 / (depth if available else 1.0))
                depth_availability_list.append(available)
            R_transpose = images.world2cams[image_id, :3, :3].T
            feature_undist = images.features_undist[image_id][feature_id]
            translation = R_transpose @ feature_undist
            translations_list.append(translation)
            image_indices_list.append(image_id2idx[image_id])
            point_indices_list.append(track_id)
    
    # Convert to tensors
    translations = torch.tensor(np.array(translations_list), dtype=torch.float64, device=device)
    image_indices = torch.tensor(np.array(image_indices_list), dtype=torch.int32, device=device)
    point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=device)
    is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                 for idx in registered_indices], 
                                 dtype=torch.bool, device=device)
    
    if use_depths:
        scales = torch.tensor(depth_values_list, dtype=torch.float64, device=device).unsqueeze(1)
        depth_availability = torch.tensor(depth_availability_list, dtype=torch.bool, device=device).unsqueeze(1)
        scale_indices = torch.where(depth_availability == 1)[0]
    else:
        scales = torch.ones(len(translations_list), 1, dtype=torch.float64, device=device)
        scale_indices = None
    
    return (image_translations, points_3d, scales, scale_indices, 
            translations, image_indices, point_indices, is_calibrated, image_idx2id)


def _prepare_multi_rig_data(cameras, images, tracks, device, min_observations, use_depths, use_fixed_rel_poses):
    """Prepare all data for multi-camera rig optimization."""
    # Filter tracks by observation count
    valid_mask = np.array([tracks.observations[i].shape[0] >= min_observations 
                           for i in range(len(tracks))])
    tracks.filter_by_mask(valid_mask)
    
    # Filter images that have no tracks
    image_used = np.zeros(len(images), dtype=bool)
    for track_id in range(len(tracks)):
        unique_image_ids = np.unique(tracks.observations[track_id][:, 0])
        image_used[unique_image_ids] = True
        if all(image_used):
            break
    images.is_registered[:] = images.is_registered & image_used
    
    # Build image and rig mappings
    image_id2idx = {}
    image_idx2id = {}
    registered_indices = images.get_registered_indices()
    for idx, image_id in enumerate(registered_indices):
        image_id2idx[image_id] = idx
        image_idx2id[idx] = image_id
    
    # Build rig group mappings
    registered_mask = images.get_registered_mask()
    group_registered = np.zeros(images.rig_groups.shape[0], dtype=bool)
    for group_idx in range(images.rig_groups.shape[0]):
        group_registered[group_idx] = any(
            registered_mask[img_id] for img_id in images.rig_groups[group_idx] if img_id != -1
        )
    
    registered_group_indices = np.where(group_registered)[0]
    row_images = {}
    image_group_idx = {}
    image_member_idx = {}
    
    for new_gid, group_idx in enumerate(registered_group_indices):
        imgs = images.rig_groups[group_idx].tolist()
        row_images[new_gid] = imgs
        for cidx, img_id in enumerate(imgs):
            if img_id != -1:
                image_group_idx[img_id] = new_gid
            image_member_idx[img_id] = cidx
    
    # Prepare tensors
    points_3d = torch.tensor(tracks.xyzs, dtype=torch.float64, device=device)
    
    # Prepare pose tensors
    ref_poses = pp.mat2SE3(images.ref_poses[registered_group_indices]).tensor().to(device).to(torch.float64)
    rel_poses = pp.mat2SE3(images.fixed_rel_poses if use_fixed_rel_poses else images.rel_poses).tensor().to(device).to(torch.float64)
    ref_trans = ref_poses[:, :3]
    ref_rots = ref_poses[:, 3:]
    rel_trans = rel_poses[:, :3]
    rel_rots = rel_poses[:, 3:]

    # Collect observation data
    feature_undist_list = []
    image_group_indices_list = []
    image_member_indices_list = []
    point_indices_list = []
    depth_values_list = []
    depth_availability_list = []

    for track_id in range(len(tracks)):
        for image_id, feature_id in tracks.observations[track_id]:
            if not images.is_registered[image_id]:
                continue
            if use_depths:
                depth = images.depths[image_id][feature_id]
                available = depth > 0
                depth_values_list.append(1.0 / (depth if available else 1.0))
                depth_availability_list.append(available)
            feature_undist = images.features_undist[image_id][feature_id]
            feature_undist_list.append(feature_undist)
            
            gid = image_group_idx[image_id]
            midx = image_member_idx[image_id]
            image_group_indices_list.append(gid)
            image_member_indices_list.append(midx)
            point_indices_list.append(track_id)

    # Convert to tensors
    feature_undist = torch.tensor(np.array(feature_undist_list), dtype=torch.float64, device=device)
    grouping_indices = torch.tensor(np.array(list(zip(image_group_indices_list, image_member_indices_list))), 
                                   dtype=torch.int32, device=device)
    point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=device)
    is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                 for idx in registered_indices], 
                                 dtype=torch.bool, device=device)
    
    if use_depths:
        scales = torch.tensor(depth_values_list, dtype=torch.float64, device=device).unsqueeze(1)
        depth_availability = torch.tensor(depth_availability_list, dtype=torch.bool, device=device).unsqueeze(1)
        scale_indices = torch.where(depth_availability == 1)[0]
    else:
        scales = torch.ones(len(feature_undist_list), 1, dtype=torch.float64, device=device)
        scale_indices = None
    
    return (points_3d, scales, ref_trans, rel_trans, ref_rots, rel_rots, scale_indices,
            feature_undist, grouping_indices, point_indices, is_calibrated,
            registered_group_indices)


def _run_optimization(optimizer, input_dict, max_iterations, function_tolerance, 
                     visualizer, update_fn, update_args, vis_name):
    """Run optimization loop with convergence checking."""
    window_size = 4
    loss_history = []
    progress_bar = tqdm.trange(max_iterations)
    
    for _ in progress_bar:
        loss = optimizer.step(input_dict)
        loss_history.append(loss.item())
        
        if len(loss_history) >= 2 * window_size:
            avg_recent = np.mean(loss_history[-window_size:])
            avg_previous = np.mean(loss_history[-2*window_size:-window_size])
            improvement = (avg_previous - avg_recent) / avg_previous
            if abs(improvement) < function_tolerance:
                break
        
        progress_bar.set_postfix({"loss": loss.item()})
        
        if visualizer:
            update_fn(*update_args)
            visualizer.add_step(*update_args[:3], vis_name)
    
    progress_bar.close()
    update_fn(*update_args)


class TorchGP():
    def __init__(self, visualizer=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.visualizer = visualizer
        
    def InitializeRandomPositions(self, cameras, images, tracks, is_multi=False, use_fixed_rel_poses=False):
        scene_scale = 100
        if is_multi:
            images.ref_poses[:, :3, 3] = scene_scale * np.random.uniform(-1, 1, (images.ref_poses.shape[0], 3))
            if use_fixed_rel_poses:
                images.rel_poses = images.fixed_rel_poses.copy()
            else:
                num_columns = images.rig_groups.shape[1]
                images.rel_poses[:, :3, 3] = np.zeros((num_columns, 3))
            Rig2Single(images)
        else:
            images.world2cams[:, :3, 3] = scene_scale * np.random.uniform(-1, 1, (len(images), 3))

        # Batch initialize track positions
        tracks.xyzs[:] = scene_scale * np.random.uniform(-1, 1, (len(tracks), 3))
        tracks.is_initialized[:] = True

    def Optimize(self, cameras, images, tracks, GLOBAL_POSITIONER_OPTIONS, use_depths=False, is_multi=False, use_fixed_rel_poses=False):
        if is_multi:
            self.OptimizeMulti(cameras, images, tracks, GLOBAL_POSITIONER_OPTIONS, use_depths=use_depths, use_fixed_rel_poses=use_fixed_rel_poses)
        else:
            self.OptimizeSingle(cameras, images, tracks, GLOBAL_POSITIONER_OPTIONS, use_depths=use_depths)

    def OptimizeSingle(self, cameras, images, tracks, GLOBAL_POSITIONER_OPTIONS, use_depths=False):
        cost_fn = pairwise_cost

        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, image_idx2id, image_translations):
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            image_translations_np = image_translations.detach().cpu().numpy()
            for idx in range(image_translations_np.shape[0]):
                image_id = image_idx2id[idx]
                images.world2cams[image_id, :3, 3] = image_translations_np[idx]
            
            for image_id in range(len(images)):
                R = images.world2cams[image_id, :3, :3]
                t = images.world2cams[image_id, :3, 3]
                images.world2cams[image_id, :3, 3] = -(R @ t)

        # Prepare all data
        (image_translations, points_3d, scales, scale_indices, 
         translations, image_indices, point_indices, is_calibrated, image_idx2id) = _prepare_single_camera_data(
            cameras, images, tracks, self.device, 
            GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'], use_depths
        )

        # Create model and optimizer
        model = PairwiseSingleCameraModel(image_translations, points_3d, scales, cost_fn, scale_indices=scale_indices)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e8, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "translations": translations,
            "image_indices": image_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
        }

        # Run optimization
        _run_optimization(
            optimizer, input, 
            GLOBAL_POSITIONER_OPTIONS['max_num_iterations'],
            GLOBAL_POSITIONER_OPTIONS['function_tolerance'],
            self.visualizer, update,
            (cameras, images, tracks, points_3d, image_idx2id, image_translations),
            "global_positioning"
        )

    def OptimizeMulti(self, cameras, images, tracks, GLOBAL_POSITIONER_OPTIONS, use_depths=False, use_fixed_rel_poses=False):
        cost_fn = pairwise_cost
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   ref_trans, rel_trans, registered_group_indices):
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            # Directly write back optimized translations to poses
            ref_trans_np = ref_trans.detach().cpu().numpy()
            rel_trans_np = rel_trans.detach().cpu().numpy()
            images.ref_poses[registered_group_indices, :3, 3] = ref_trans_np
            images.rel_poses[:, :3, 3] = rel_trans_np
            Rig2Single(images)
        
        # Prepare all data
        (points_3d, scales, ref_trans, rel_trans, ref_rots, rel_rots, scale_indices,
         feature_undist, grouping_indices, point_indices, is_calibrated,
         registered_group_indices) = _prepare_multi_rig_data(
            cameras, images, tracks, self.device,
            GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'], use_depths, use_fixed_rel_poses
        )

        # Create model and optimizer
        if use_fixed_rel_poses:
            # Use FixedRel model when rel_trans are fixed (passed via forward)
            model = PairwiseMultiRigModelFixedRel(points_3d, scales, ref_trans, 
                                                 cost_fn, scale_indices=scale_indices)
        else:
            # Use regular model when rel_trans are optimized (nn.Parameter)
            model = PairwiseMultiRigModel(points_3d, scales, ref_trans, rel_trans,
                                         cost_fn, scale_indices=scale_indices)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e8, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "feature_undist": feature_undist,
            "grouping_indices": grouping_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
            "ref_rots": ref_rots,
            "rel_rots": rel_rots
        }
        if use_fixed_rel_poses:
            # Pass rel_trans via input dict for FixedRel model
            input["rel_trans"] = rel_trans
        
        # Run optimization
        _run_optimization(
            optimizer, input,
            GLOBAL_POSITIONER_OPTIONS['max_num_iterations'],
            GLOBAL_POSITIONER_OPTIONS['function_tolerance'],
            self.visualizer, update,
            (cameras, images, tracks, points_3d, ref_trans, rel_trans, registered_group_indices),
            "global_positioning"
        )