import numpy as np
from scipy.spatial.transform import Rotation as R

from instantsfm.processors.image_undistortion import UndistortImages
from instantsfm.processors.track_filter import FilterTracksByReprojection, FilterTracksTriangulationAngle
from instantsfm.processors.bundle_adjustment import TorchBA
from instantsfm.utils.union_find import UnionFind
from instantsfm.scene.defs import CameraModelId, get_camera_model_info
from instantsfm.utils.cost_function import reproject_funcs_no_depth

import torch
from bae.utils.ba import rotate_quat

EPSILON = 1e-7

def complete_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS):
    """
    Args:
      cameras : list of `Camera` objects (indexed by cam_id).
      images  : list of image objects, each with:
                - .cam_id => references which camera it uses
                - .features[feature_id] => 2D [x,y] for each feature
                - .correspondences[feature_id] => list of (other_image_id, other_feature_id)
      tracks  : dict of track objects, each with:
                  - .xyz => (3,) world coordinate
                  - .observations => np.array of (image_id, feature_id)
      tracks_orig: dict of track objects, same as `tracks` but with more observations and points
      TRIANGULATOR_OPTIONS: dict containing options.
    Returns:
      num_completed: number of new (image_id, feature_id) observations changed across all tracks.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    reproj_threshold = TRIANGULATOR_OPTIONS['complete_max_reproj_error']
    camera_model = cameras[0].model_id # Assume all cameras have the same model
    camera_model_info = get_camera_model_info(camera_model)
    try:
        cost_fn = reproject_funcs_no_depth[camera_model.value]
    except:
        raise NotImplementedError("Unsupported camera model")

    candidate_observed_features = []  # shape: [B, 2] 2D coords in pixel space
    candidate_track_indices     = []  # shape: [B] which row in pairwise_module.points_3d
    candidate_obs_info          = []  # shape: [B, 2] (image_id, feature_id) corresponding to observed_feature

    track_id2idx = {int(track_id): idx for idx, track_id in enumerate(tracks.ids)}

    for track_id, track_obs in tracks_orig.items():
        if track_id not in track_id2idx:
            continue

        candidate_observed_features.extend([images[img_id].features[feat_id] for img_id, feat_id in track_obs])
        candidate_track_indices.extend([track_id2idx[track_id] for _ in range(len(track_obs))])
        candidate_obs_info.extend(track_obs)

    if not candidate_observed_features:
        return 0

    # Convert to torch tensors
    observed_features_tensor = torch.tensor(np.array(candidate_observed_features), dtype=torch.float64, device=device) # [n, 2]
    point_indices_tensor = torch.tensor(candidate_track_indices, dtype=torch.int32, device=device) # [n]
    obs_info_tensor = torch.tensor(np.array(candidate_obs_info), dtype=torch.int32, device=device) # [n, 2]
    camera_indices_tensor = obs_info_tensor[:, 0] # [n]

    camera_model = cameras[0].model_id # Assume all cameras have the same model
    # Batch build camera params from Images container
    camera_params_list = []
    for img_id in range(len(images)):
        world2cam = images.world2cams[img_id]
        cam_id = images.cam_ids[img_id]
        camera_params_list.append(torch.cat((
            torch.tensor(world2cam[:3, 3]),
            torch.tensor(R.from_matrix(world2cam[:3, :3]).as_quat()),
            torch.tensor(cameras[cam_id].params)
        )))
    camera_params = torch.stack(camera_params_list, dim=0).to(device).to(torch.float64)
    # Batch extract track positions
    points_3d = torch.tensor(tracks.xyzs, device=device, dtype=torch.float64)

    # Indexing
    points_3d = points_3d[point_indices_tensor]
    camera_params = camera_params[camera_indices_tensor]
    pp_indices = torch.tensor(camera_model_info['pp'], device=device) + 7 # add 7 for translation and rotation
    camera_pps = camera_params[..., pp_indices]
    all_indices = torch.arange(camera_params.shape[1], device=device)
    remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=device)
    camera_params = camera_params[..., remaining_indices]

    camera_extrinsics = camera_params[..., :7]
    camera_intrinsics = camera_params[..., 7:]

    points_proj = rotate_quat(points_3d, camera_extrinsics)
    valid_mask = points_proj[..., 2] > EPSILON # filter out points behind the camera

    errors = cost_fn(points_3d, camera_extrinsics, camera_intrinsics, camera_pps)
    errors -= observed_features_tensor
    errors = torch.norm(errors, dim=-1)

    # Filter results by threshold
    passing_mask = (errors <= reproj_threshold)
    passing_mask = passing_mask & valid_mask
    if not torch.any(passing_mask):
        return 0
    obs_info = obs_info_tensor[passing_mask].detach().cpu().numpy()
    point_indices_tensor = point_indices_tensor[passing_mask]

    # For each passing candidate, update track observations
    split_indices = torch.nonzero(torch.diff(point_indices_tensor)).squeeze(1) + 1
    split_indices = torch.cat([torch.tensor([0]).to(split_indices.device), 
                               split_indices, 
                               torch.tensor([point_indices_tensor.shape[0]]).to(split_indices.device)]).detach().cpu().tolist()

    num_completed = 0
    for i in range(len(split_indices) - 1):
        track_idx = point_indices_tensor[split_indices[i]].item()
        old_num_obs = tracks.observations[track_idx].shape[0]
        new_obs = obs_info[split_indices[i]:split_indices[i+1]]
        num_completed += abs((split_indices[i+1] - split_indices[i]) - old_num_obs)
        tracks.observations[track_idx] = new_obs

    return num_completed

def merge_tracks(cameras, images, tracks, TRIANGULATOR_OPTIONS):
    """
    Attempts to merge 3D point tracks if their merged 3D point produces
    acceptable reprojection errors in all observations. 

    Parameters:
        cameras: list of camera objects.
        images: list of image objects.
                Each image is expected to have:
                  - world2cam: a (4,4) NumPy array.
                  - cam_id: index or key for accessing its camera.
        tracks: dict mapping track_id to a track object.
                Each track object must have:
                  - xyz: NumPy array of shape (3,).
                  - observations: list of tuples (image_id, feature_id).
        TRIANGULATOR_OPTIONS: dict containing options.

    Returns:
        Total number of merged observations.
    """
    max_squared_reproj_error = TRIANGULATOR_OPTIONS['merge_max_reproj_error'] ** 2

    # Batch extract xyz positions
    xyz_points = tracks.xyzs
    track_idx2id = {idx: idx for idx in range(len(tracks))}

    k = min(3, len(tracks))
    index = faiss.IndexFlatL2(3)
    index.add(xyz_points)
    distances, indices = index.search(xyz_points, k)

    candidate_pairs = []
    
    for i in range(len(tracks)):
        for j in range(1, len(indices[i])): # skip the first one, which is the track itself
            source_idx = i
            target_idx = indices[i][j]
            candidate_pairs.append((source_idx, target_idx))

    source_idx, target_idx = zip(*[(source_idx, target_idx) for source_idx, target_idx in candidate_pairs])
    source_id, target_id = [track_idx2id[idx] for idx in source_idx], [track_idx2id[idx] for idx in target_idx]

    uf = UnionFind()
    for track_id in range(len(tracks)):
        uf.Find(track_id)

    def try_merge_pair(track_id1, track_id2):
        # merge track1 into track2
        track_actual_id1, track_actual_id2 = uf.Find(track_id1), uf.Find(track_id2)
        if track_actual_id1 == track_actual_id2:
            return False, None
        
        track1_xyz = tracks.xyzs[track_actual_id1]
        track2_xyz = tracks.xyzs[track_actual_id2]
        track1_obs = tracks.observations[track_actual_id1]
        track2_obs = tracks.observations[track_actual_id2]

        weight1 = track1_obs.shape[0]
        weight2 = track2_obs.shape[0]
        merged_xyz = (weight1 * track1_xyz + weight2 * track2_xyz) / (weight1 + weight2)

        all_obs = np.concatenate([track1_obs, track2_obs], axis=0)

        for image_id, feature_id in all_obs:
            world2cam = images.world2cams[image_id]
            pt_calc = world2cam[:3, :3] @ merged_xyz + world2cam[:3, 3]
            if pt_calc[2] < EPSILON:
                return False, None
            feature = images.features[image_id][feature_id]
            cam = cameras[images.cam_ids[image_id]]
            pt_reproj = cam.cam2img(pt_calc)
            sq_reprojection_error = np.sum((pt_reproj - feature) ** 2)
            if sq_reprojection_error > max_squared_reproj_error:
                return False, None
        return True, (merged_xyz, all_obs)
            
    total_merged = 0

    for src, tgt in zip(source_id, target_id):
        merged, result = try_merge_pair(src, tgt)
        if merged:
            uf.Union(src, tgt)
            total_merged += len(tracks.observations[src])
            # save new track into target track (align behavior with union find)
            final_tgt = uf.Find(tgt)
            tracks.xyzs[final_tgt] = result[0]
            tracks.observations[final_tgt] = result[1]

    # Mark deleted tracks and filter in-place
    valid_mask = np.array([uf.Find(track_id) == track_id for track_id in range(len(tracks))], dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    
    # Filter tracks in-place to maintain reference semantics
    tracks.filter_by_mask(valid_mask)

    return total_merged

def filter_points(cameras, images, tracks, TRIANGULATOR_OPTIONS):
    num_filtered = 0
    num_filtered += FilterTracksByReprojection(cameras, images, tracks, TRIANGULATOR_OPTIONS['filter_max_reproj_error'])
    num_filtered += FilterTracksTriangulationAngle(cameras, images, tracks, TRIANGULATOR_OPTIONS['filter_min_tri_angle'])
    return num_filtered

def complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS):
    num_completed_observations = complete_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS)
    print('Number of completed observations:', num_completed_observations)
    num_merged_observations = 0
    # TODO: Implement a better merge later, current version of merge does not have a good result and is not used in the pipeline
    # num_merged_observations = merge_tracks(cameras, images, tracks, TRIANGULATOR_OPTIONS)
    # print('Number of merged observations:', num_merged_observations)
    return num_completed_observations + num_merged_observations

def RetriangulateTracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS, BUNDLE_ADJUSTER_OPTIONS):
    # record status of images
    image_registered = images.is_registered.copy()

    # add new tracks
    '''P = []
    for image_id in range(len(images)):
        cam = cameras[images.cam_ids[image_id]]
        Rt = images.world2cams[image_id]
        K = cam.get_K()
        P.append(K @ Rt[:3])

    for track_id, track_obs in tracks_orig.items():
        if track_id not in tracks and track_obs.shape[0] == 2:
            # test: opencv triangulation
            image_idx1, image_idx2 = track_obs[0, 0], track_obs[1, 0]
            if not image_registered[image_idx1] or not image_registered[image_idx2]:
                continue
            P1, P2 = P[image_idx1], P[image_idx2]
            points1 = images.features[image_idx1][track_obs[0, 1]]
            points2 = images.features[image_idx2][track_obs[1, 1]]
            point4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
            point3D = point4D[:3] / point4D[3]
            # Note: This code is commented out and not used
            # Would need to use tracks.append() or similar if re-enabled
            '''
    
    complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS)

    for i in range(TRIANGULATOR_OPTIONS['ba_global_max_refinements']):
        print(f'Running bundle adjustment iteration {i+1} / {TRIANGULATOR_OPTIONS["ba_global_max_refinements"]}') 
        ba_engine = TorchBA()
        LOCAL_BUNDLE_ADJUSTER_OPTIONS = BUNDLE_ADJUSTER_OPTIONS.copy()
        LOCAL_BUNDLE_ADJUSTER_OPTIONS['optimize_poses'] = False
        ba_engine.Solve(cameras, images, tracks, LOCAL_BUNDLE_ADJUSTER_OPTIONS)
        num_changed_observations = 0
        num_changed_observations += abs(complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS))
        num_changed_observations += filter_points(cameras, images, tracks, TRIANGULATOR_OPTIONS)
        changed_percentage = num_changed_observations / len(tracks)
        if changed_percentage < TRIANGULATOR_OPTIONS['ba_global_max_refinement_change']:
            break
    
    # restore status of images
    images.is_registered[:] = image_registered
