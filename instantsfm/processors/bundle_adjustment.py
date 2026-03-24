import numpy as np
import tqdm

from instantsfm.scene.defs import get_camera_model_info
from instantsfm.scene.rig_utils import Rig2Single
from instantsfm.utils.cost_function import reproject_funcs, reproject_funcs_no_depth
from instantsfm.utils.optimization_models import (
    ReprojectionModel, ReprojectionModelWithDepth,
    ReprojectionMultiRigModel, ReprojectionMultiRigModelWithDepth,
    ReprojectionMultiRigModelFixedRel, ReprojectionMultiRigModelWithDepthFixedRel
)

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Huber
from bae.utils.pysolvers import PCG
from bae.utils.ba import rotate_quat
from bae.optim import LM
from bae.autograd.function import TrackingTensor


def _prepare_single_camera_data(cameras, images, tracks, camera_model_info, device, min_observations, use_depths):
    """Prepare all data for single camera bundle adjustment."""
    # Filter tracks by observation count
    valid_tracks_mask = np.array([tracks.observations[i].shape[0] >= min_observations 
                                  for i in range(len(tracks))], dtype=bool)
    tracks.filter_by_mask(valid_tracks_mask)
    
    # Filter images that have no tracks
    image_used = np.zeros(len(images), dtype=bool)
    for i in range(len(tracks)):
        unique_image_ids = np.unique(tracks.observations[i][:, 0])
        image_used[unique_image_ids] = True
        if all(image_used):
            break
    images.is_registered[:] = image_used
    
    # Build image ID mappings
    image_id2idx = {}
    image_idx2id = {}
    registered_mask = images.get_registered_mask()
    for image_id in range(len(images)):
        if not registered_mask[image_id]:
            continue
        image_id2idx[image_id] = len(image_id2idx)
        image_idx2id[len(image_idx2id)] = image_id

    # Filter cameras by usage
    camera_used = np.zeros(len(cameras), dtype=bool)
    registered_cam_ids = images.cam_ids[registered_mask]
    camera_used[registered_cam_ids] = True
    camera_id2idx = {}
    camera_idx2id = {}
    for camera_id, camera in enumerate(cameras):
        if not camera_used[camera_id]:
            continue
        camera_id2idx[camera_id] = len(camera_id2idx)
        camera_idx2id[len(camera_idx2id)] = camera_id

    # Prepare pose and intrinsic tensors
    registered_indices = images.get_registered_indices()
    image_extrs_list = [pp.mat2SE3(images.world2cams[idx]).tensor() for idx in registered_indices]
    image_extrs = torch.stack(image_extrs_list, dim=0).to(device).to(torch.float64)
    
    camera_intrs_list = [torch.tensor(camera.params) for idx, camera in enumerate(cameras) if camera_used[idx]]
    camera_intrs = torch.stack(camera_intrs_list, dim=0).to(device).to(torch.float64)

    # Separate principal points (not optimized)
    pp_indices = torch.tensor(camera_model_info['pp'], device=device)
    camera_pps = camera_intrs[..., pp_indices]
    all_indices = torch.arange(camera_intrs.shape[1], device=device)
    remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=device)
    camera_intrs = camera_intrs[..., remaining_indices]
    
    # Prepare track positions
    points_3d = torch.tensor(tracks.xyzs, device=device, dtype=torch.float64)

    # Collect observation data
    points_2d_list = []
    image_indices_list = []
    camera_indices_list = []
    point_indices_list = []
    depths_ref_list = []
    
    for track_id in range(len(tracks)):
        track_obs = tracks.observations[track_id]
        for image_id, feature_id in track_obs:
            if not images.is_registered[image_id]:
                continue
            point2D = images.features[image_id][feature_id]
            points_2d_list.append(point2D)
            image_indices_list.append(image_id2idx[image_id])
            camera_indices_list.append(camera_id2idx[images.cam_ids[image_id]])
            point_indices_list.append(track_id)
            
            if use_depths:
                depth_ref = images.depths[image_id][feature_id]
                depths_ref_list.append(depth_ref if depth_ref > 0 else 1e8)

    # Convert to tensors
    points_2d = torch.tensor(np.array(points_2d_list), dtype=torch.float64, device=device)
    image_indices = torch.tensor(np.array(image_indices_list), dtype=torch.int32, device=device)
    camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32, device=device)
    point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=device)
    
    depths_ref = None
    if use_depths:
        depths_ref = torch.tensor(np.array(depths_ref_list), dtype=torch.float64, device=device)
        eps = 1e-6
        depths_ref = 1.0 / (depths_ref + eps)
    
    return (image_extrs, camera_intrs, points_3d, camera_pps, remaining_indices, pp_indices,
            points_2d, image_indices, camera_indices, point_indices, depths_ref,
            image_idx2id, camera_idx2id)


def _prepare_multi_rig_data(cameras, images, tracks, camera_model_info, device, min_observations, use_depths):
    """Prepare all data for multi-rig bundle adjustment."""
    # Filter tracks by observation count
    valid_tracks_mask = np.array([tracks.observations[i].shape[0] >= min_observations 
                                  for i in range(len(tracks))], dtype=bool)
    tracks.filter_by_mask(valid_tracks_mask)

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
    
    # Prepare pose tensors
    ref_poses = pp.mat2SE3(images.ref_poses[registered_group_indices]).to(device).to(torch.float64)
    rel_poses = pp.mat2SE3(images.rel_poses).to(device).to(torch.float64)

    # Prepare camera intrinsics (all cameras used in rig)
    camera_intrs_list = []
    for idx, camera in enumerate(cameras):
        cam_params = torch.tensor(camera.params)
        camera_intrs_list.append(cam_params)
    camera_intrs = torch.stack(camera_intrs_list, dim=0).to(device).to(torch.float64)
    
    # Separate principal points
    pp_indices = torch.tensor(camera_model_info['pp'], device=device)
    camera_pps = camera_intrs[..., pp_indices]
    all_indices = torch.arange(camera_intrs.shape[1], device=device)
    remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=device)
    camera_intrs = camera_intrs[..., remaining_indices]
    
    # Prepare track positions
    points_3d = torch.tensor(tracks.xyzs, device=device, dtype=torch.float64)

    # Collect observation data
    points_2d_list = []
    camera_indices_list = []
    image_group_indices_list = []
    image_member_indices_list = []
    point_indices_list = []
    depths_ref_list = []
    
    for track_id in range(len(tracks)):
        track_obs = tracks.observations[track_id]
        for image_id, feature_id in track_obs:
            if not images.is_registered[image_id]:
                continue
            point2D = images.features[image_id][feature_id]
            points_2d_list.append(point2D)
            camera_indices_list.append(images.cam_ids[image_id])

            gid = image_group_idx[image_id]
            midx = image_member_idx[image_id]
            image_group_indices_list.append(gid)
            image_member_indices_list.append(midx)
            point_indices_list.append(track_id)
            
            if use_depths:
                depth_ref = images.depths[image_id][feature_id]
                depths_ref_list.append(depth_ref if depth_ref > 0 else 1e8)

    # Convert to tensors
    points_2d = torch.tensor(np.array(points_2d_list), dtype=torch.float64, device=device)
    camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32, device=device)
    grouping_indices = torch.tensor(np.array(list(zip(image_group_indices_list, image_member_indices_list))), 
                                   dtype=torch.int32, device=device)
    point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=device)
    
    depths_ref = None
    if use_depths:
        depths_ref = torch.tensor(np.array(depths_ref_list), dtype=torch.float64, device=device)
        eps = 1e-6
        depths_ref = 1.0 / (depths_ref + eps)
    
    return (camera_intrs, points_3d, ref_poses, rel_poses, camera_pps, remaining_indices, pp_indices,
            points_2d, camera_indices, grouping_indices, point_indices, depths_ref,
            image_group_idx, image_member_idx)


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
            if loss_history[-1] == loss_history[-2]:
                break
        
        progress_bar.set_postfix({"loss": loss.item()})
        
        if visualizer:
            update_fn(*update_args)
            visualizer.add_step(*update_args[:3], vis_name)
    
    progress_bar.close()
    update_fn(*update_args)


class TorchBA():
    def __init__(self, visualizer=None, device="cuda:0"):
        super().__init__()
        self.device = device
        self.visualizer = visualizer

    def Solve(
        self,
        cameras,
        images,
        tracks,
        BUNDLE_ADJUSTER_OPTIONS,
        use_depths=False,
        is_multi=False,
        use_fixed_rel_poses=False,
        optimize_intrinsics=True,
    ):
        if is_multi:
            self.SolveMulti(
                cameras,
                images,
                tracks,
                BUNDLE_ADJUSTER_OPTIONS,
                use_depths=use_depths,
                use_fixed_rel_poses=use_fixed_rel_poses,
                optimize_intrinsics=optimize_intrinsics,
            )
        else:
            self.SolveSingle(
                cameras,
                images,
                tracks,
                BUNDLE_ADJUSTER_OPTIONS,
                use_depths=use_depths,
                optimize_intrinsics=optimize_intrinsics,
            )
    
    def SolveSingle(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS, use_depths=False, optimize_intrinsics=True):
        self.camera_model = cameras[0].model_id
        self.camera_model_info = get_camera_model_info(self.camera_model)
        try:
            cost_fn = reproject_funcs[self.camera_model.value] if use_depths else reproject_funcs_no_depth[self.camera_model.value]
        except:
            raise NotImplementedError("Unsupported camera model")
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   image_idx2id, camera_idx2id,
                   image_extrs, camera_intrs, camera_pps, remaining_indices, pp_indices):
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            image_extrs_np = pp.SE3(image_extrs).matrix().cpu().numpy()
            for idx in range(image_extrs_np.shape[0]):
                image_id = image_idx2id[idx]
                images.world2cams[image_id] = image_extrs_np[idx]
            
            # Reconstruct camera parameters
            max_rem = int(torch.max(remaining_indices).item()) if remaining_indices.numel() > 0 else -1
            max_pp = int(torch.max(pp_indices).item()) if pp_indices.numel() > 0 else -1
            full_len = max(max_rem, max_pp) + 1
            camera_params_full = torch.zeros((camera_intrs.shape[0], full_len), 
                                           dtype=camera_intrs.dtype, device=camera_intrs.device)
            camera_params_full[:, remaining_indices] = camera_intrs
            camera_params_full[:, pp_indices] = camera_pps
            camera_params_np = camera_params_full.detach().cpu().numpy()
            for idx in range(camera_params_np.shape[0]):
                cam = cameras[camera_idx2id[idx]]
                cam.set_params(camera_params_np[idx])
        
        # Prepare all data
        (image_extrs, camera_intrs, points_3d, camera_pps, remaining_indices, pp_indices,
         points_2d, image_indices, camera_indices, point_indices, depths_ref,
         image_idx2id, camera_idx2id) = _prepare_single_camera_data(
            cameras, images, tracks, self.camera_model_info, self.device,
            BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track'], use_depths
        )

        # Create model
        if use_depths:
            model = ReprojectionModelWithDepth(
                image_extrs,
                camera_intrs,
                points_3d,
                cost_fn,
                depth_weight=BUNDLE_ADJUSTER_OPTIONS['depth_weight'],
                optimize_intrinsics=optimize_intrinsics,
            )
        else:
            model = ReprojectionModel(
                image_extrs,
                camera_intrs,
                points_3d,
                cost_fn,
                optimize_intrinsics=optimize_intrinsics,
            )
        
        # Create optimizer
        strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        # Prepare input dict
        input = {
            "points_2d": points_2d,
            "image_indices": image_indices,
            "camera_indices": camera_indices,
            "point_indices": point_indices,
            "camera_pps": camera_pps,
        }
        if use_depths:
            input["depths_ref"] = depths_ref

        # Run optimization
        _run_optimization(
            optimizer, input,
            BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'],
            BUNDLE_ADJUSTER_OPTIONS['function_tolerance'],
            self.visualizer, update,
            (cameras, images, tracks, points_3d, image_idx2id, camera_idx2id, 
             image_extrs, camera_intrs, camera_pps, remaining_indices, pp_indices),
            "bundle_adjustment"
        )

    def SolveMulti(
        self,
        cameras,
        images,
        tracks,
        BUNDLE_ADJUSTER_OPTIONS,
        use_depths=False,
        use_fixed_rel_poses=False,
        optimize_intrinsics=True,
    ):
        self.camera_model = cameras[0].model_id
        self.camera_model_info = get_camera_model_info(self.camera_model)
        try:
            cost_fn = reproject_funcs[self.camera_model.value] if use_depths else reproject_funcs_no_depth[self.camera_model.value]
        except:
            raise NotImplementedError("Unsupported camera model")
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   camera_intrs, camera_pps, remaining_indices, pp_indices,
                   ref_poses, rel_poses):
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()

            registered_mask = images.get_registered_mask()
            group_registered = np.zeros(images.rig_groups.shape[0], dtype=bool)
            for group_idx in range(images.rig_groups.shape[0]):
                group_registered[group_idx] = any(
                    registered_mask[img_id] for img_id in images.rig_groups[group_idx] if img_id != -1
                )
            registered_group_indices = np.where(group_registered)[0]
            
            ref_poses_mats = pp.SE3(ref_poses).matrix().cpu().numpy()
            rel_poses_mats = pp.SE3(rel_poses).matrix().cpu().numpy()
            images.ref_poses[registered_group_indices] = ref_poses_mats
            images.rel_poses = rel_poses_mats
            Rig2Single(images)
            
            # Reconstruct camera parameters
            max_rem = int(torch.max(remaining_indices).item()) if remaining_indices.numel() > 0 else -1
            max_pp = int(torch.max(pp_indices).item()) if pp_indices.numel() > 0 else -1
            full_len = max(max_rem, max_pp) + 1
            camera_params_full = torch.zeros((camera_intrs.shape[0], full_len), 
                                           dtype=camera_intrs.dtype, device=camera_intrs.device)
            camera_params_full[:, remaining_indices] = camera_intrs
            camera_params_full[:, pp_indices] = camera_pps
            camera_params_np = camera_params_full.detach().cpu().numpy()
            for idx, cam in enumerate(cameras):
                cam.set_params(camera_params_np[idx])

        # Prepare all data
        (camera_intrs, points_3d, ref_poses, rel_poses, camera_pps, remaining_indices, pp_indices,
         points_2d, camera_indices, grouping_indices, point_indices, depths_ref,
         image_group_idx, image_member_idx) = _prepare_multi_rig_data(
            cameras, images, tracks, self.camera_model_info, self.device,
            BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track'], use_depths
        )
        
        # Create model
        if use_fixed_rel_poses:
            # Use FixedRel models when rel_poses are fixed (passed via forward)
            if use_depths:
                model = ReprojectionMultiRigModelWithDepthFixedRel(
                    camera_intrs,
                    points_3d,
                    ref_poses,
                    cost_fn,
                    depth_weight=BUNDLE_ADJUSTER_OPTIONS['depth_weight'],
                    optimize_intrinsics=optimize_intrinsics,
                )
            else:
                model = ReprojectionMultiRigModelFixedRel(
                    camera_intrs,
                    points_3d,
                    ref_poses,
                    cost_fn,
                    optimize_intrinsics=optimize_intrinsics,
                )
        else:
            # Use regular models when rel_poses are optimized (nn.Parameter)
            if use_depths:
                model = ReprojectionMultiRigModelWithDepth(
                    camera_intrs,
                    points_3d,
                    ref_poses,
                    rel_poses,
                    cost_fn,
                    depth_weight=BUNDLE_ADJUSTER_OPTIONS['depth_weight'],
                    optimize_intrinsics=optimize_intrinsics,
                )
            else:
                model = ReprojectionMultiRigModel(
                    camera_intrs,
                    points_3d,
                    ref_poses,
                    rel_poses,
                    cost_fn,
                    optimize_intrinsics=optimize_intrinsics,
                )
        
        # Create optimizer
        strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        # Prepare input dict
        input = {
            "points_2d": points_2d,
            "camera_indices": camera_indices,
            "grouping_indices": grouping_indices,
            "point_indices": point_indices,
            "camera_pps": camera_pps,
        }
        if use_depths:
            input["depths_ref"] = depths_ref
        if use_fixed_rel_poses:
            # Pass rel_poses via input dict for FixedRel models
            input["rel_poses"] = rel_poses

        # Run optimization
        _run_optimization(
            optimizer, input,
            BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'],
            BUNDLE_ADJUSTER_OPTIONS['function_tolerance'],
            self.visualizer, update,
            (cameras, images, tracks, points_3d, 
             camera_intrs, camera_pps, remaining_indices, pp_indices,
             ref_poses, rel_poses),
            "bundle_adjustment"
        )
