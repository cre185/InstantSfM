import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from instantsfm.processors.image_undistortion import UndistortImages
from instantsfm.processors.track_filter import FilterTracksByReprojection, FilterTracksTriangulationAngle
from instantsfm.processors.bundle_adjustment import TorchBA
from instantsfm.utils.union_find import UnionFind
from instantsfm.scene.defs import CameraModelId, get_camera_model_info
from instantsfm.utils.cost_function import reproject_funcs_no_depth

import torch
from bae.utils.ba import rotate_quat

EPSILON = 1e-7


def _make_observation_to_track_map(tracks):
    obs_to_track = {}
    for track_idx, track_obs in enumerate(tracks.observations):
        for image_id, feature_id in track_obs:
            obs_to_track[(int(image_id), int(feature_id))] = track_idx
    return obs_to_track


def _triangulate_two_view_observation(cameras, images, obs1, obs2):
    image_id1, feature_id1 = int(obs1[0]), int(obs1[1])
    image_id2, feature_id2 = int(obs2[0]), int(obs2[1])

    world2cam1 = images.world2cams[image_id1][:3, :]
    world2cam2 = images.world2cams[image_id2][:3, :]
    feat1 = images.features_undist[image_id1][feature_id1]
    feat2 = images.features_undist[image_id2][feature_id2]
    xy1 = (feat1[:2] / max(feat1[2], EPSILON)).reshape(2, 1)
    xy2 = (feat2[:2] / max(feat2[2], EPSILON)).reshape(2, 1)

    point_h = cv2.triangulatePoints(world2cam1, world2cam2, xy1, xy2)
    if abs(point_h[3, 0]) < EPSILON:
        return None
    xyz = (point_h[:3, 0] / point_h[3, 0]).astype(np.float64)
    return xyz


def _point_reprojection_is_valid(cameras, images, xyz, image_id, feature_id, reproj_threshold):
    world2cam = images.world2cams[image_id]
    pt_calc = world2cam[:3, :3] @ xyz + world2cam[:3, 3]
    if pt_calc[2] < EPSILON:
        return False
    feature = images.features[image_id][feature_id]
    cam = cameras[images.cam_ids[image_id]]
    pt_reproj = cam.cam2img(pt_calc)
    sq_reprojection_error = np.sum((pt_reproj - feature) ** 2)
    return sq_reprojection_error <= reproj_threshold ** 2


def _track_supports_observation(cameras, images, xyz, track_obs, obs, reproj_threshold):
    image_id = int(obs[0])
    if image_id in track_obs[:, 0]:
        return False

    all_obs = np.vstack([track_obs, np.asarray(obs, dtype=np.int32).reshape(1, 2)])
    for obs_image_id, obs_feature_id in all_obs:
        if not _point_reprojection_is_valid(
            cameras, images, xyz, int(obs_image_id), int(obs_feature_id), reproj_threshold
        ):
            return False
    return True


def _triangulation_angle_is_valid(images, xyz, obs1, obs2, min_angle_deg):
    image_ids = np.array([int(obs1[0]), int(obs2[0])], dtype=np.int32)
    Rs = images.world2cams[image_ids, :3, :3]
    ts = images.world2cams[image_ids, :3, 3]
    centers = -np.einsum('nij,nj->ni', Rs.transpose(0, 2, 1), ts)
    rays = xyz.reshape(1, 3) - centers
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    if np.any(norms < EPSILON):
        return False
    rays = rays / norms
    cos_angle = float(np.clip(np.dot(rays[0], rays[1]), -1.0, 1.0))
    angle_deg = np.rad2deg(np.arccos(cos_angle))
    return angle_deg >= min_angle_deg


def _get_pair_matches(pair):
    if len(pair.inliers) > 0:
        return pair.matches[np.asarray(pair.inliers, dtype=np.int32)]
    return pair.matches


def retriangulate_underreconstructed_pairs(view_graph, cameras, images, tracks, TRIANGULATOR_OPTIONS, pair_retry_counts):
    reproj_threshold = TRIANGULATOR_OPTIONS['complete_max_reproj_error']
    re_min_ratio = TRIANGULATOR_OPTIONS.get('re_min_ratio', 0.2)
    re_max_trials = TRIANGULATOR_OPTIONS.get('re_max_trials', 1)
    min_tri_angle = TRIANGULATOR_OPTIONS.get('filter_min_tri_angle', 1.5)
    ignore_two_view_tracks = TRIANGULATOR_OPTIONS.get('ignore_two_view_tracks', True)

    obs_to_track = _make_observation_to_track_map(tracks)
    num_retriangulated_observations = 0
    next_track_id = int(tracks.ids.max()) + 1 if len(tracks) > 0 else 0

    for pair_key, pair in view_graph.image_pairs.items():
        if not pair.is_valid:
            continue
        image_id1, image_id2 = int(pair.image_id1), int(pair.image_id2)
        if not images.is_registered[image_id1] or not images.is_registered[image_id2]:
            continue

        pair_matches = _get_pair_matches(pair)
        if pair_matches.shape[0] == 0:
            continue

        num_triangulated_corrs = 0
        for feat_id1, feat_id2 in pair_matches:
            track_idx1 = obs_to_track.get((image_id1, int(feat_id1)))
            track_idx2 = obs_to_track.get((image_id2, int(feat_id2)))
            if track_idx1 is not None and track_idx1 == track_idx2:
                num_triangulated_corrs += 1

        tri_ratio = num_triangulated_corrs / float(pair_matches.shape[0])
        if tri_ratio >= re_min_ratio:
            continue
        if pair_retry_counts.get(pair_key, 0) >= re_max_trials:
            continue
        pair_retry_counts[pair_key] = pair_retry_counts.get(pair_key, 0) + 1

        for feat_id1, feat_id2 in pair_matches:
            obs1 = (image_id1, int(feat_id1))
            obs2 = (image_id2, int(feat_id2))
            track_idx1 = obs_to_track.get(obs1)
            track_idx2 = obs_to_track.get(obs2)

            if track_idx1 is not None and track_idx2 is not None:
                continue

            if track_idx1 is not None:
                if _track_supports_observation(
                    cameras, images, tracks.xyzs[track_idx1], tracks.observations[track_idx1], obs2, reproj_threshold
                ):
                    tracks.observations[track_idx1] = np.vstack(
                        [tracks.observations[track_idx1], np.asarray(obs2, dtype=np.int32).reshape(1, 2)]
                    )
                    obs_to_track[obs2] = track_idx1
                    num_retriangulated_observations += 1
                continue

            if track_idx2 is not None:
                if _track_supports_observation(
                    cameras, images, tracks.xyzs[track_idx2], tracks.observations[track_idx2], obs1, reproj_threshold
                ):
                    tracks.observations[track_idx2] = np.vstack(
                        [tracks.observations[track_idx2], np.asarray(obs1, dtype=np.int32).reshape(1, 2)]
                    )
                    obs_to_track[obs1] = track_idx2
                    num_retriangulated_observations += 1
                continue

            if ignore_two_view_tracks:
                continue

            xyz = _triangulate_two_view_observation(cameras, images, obs1, obs2)
            if xyz is None:
                continue
            if not _triangulation_angle_is_valid(images, xyz, obs1, obs2, min_tri_angle):
                continue
            if not _point_reprojection_is_valid(cameras, images, xyz, image_id1, int(feat_id1), reproj_threshold):
                continue
            if not _point_reprojection_is_valid(cameras, images, xyz, image_id2, int(feat_id2), reproj_threshold):
                continue

            tracks.append(
                id=next_track_id,
                xyz=xyz,
                is_initialized=True,
                observations=np.asarray([obs1, obs2], dtype=np.int32),
            )
            obs_to_track[obs1] = len(tracks) - 1
            obs_to_track[obs2] = len(tracks) - 1
            next_track_id += 1
            num_retriangulated_observations += 2

    return num_retriangulated_observations


def count_observations(tracks) -> int:
    return sum(track_obs.shape[0] for track_obs in tracks.observations)

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
        merged_obs = np.concatenate([tracks.observations[track_idx], new_obs], axis=0)
        merged_obs = np.unique(merged_obs, axis=0)
        num_completed += max(0, merged_obs.shape[0] - old_num_obs)
        tracks.observations[track_idx] = merged_obs

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

def filter_points(cameras, images, tracks, TRIANGULATOR_OPTIONS, use_triangulation_angle=True):
    num_observations_before = count_observations(tracks)
    FilterTracksByReprojection(cameras, images, tracks, TRIANGULATOR_OPTIONS['filter_max_reproj_error'])
    if use_triangulation_angle:
        FilterTracksTriangulationAngle(cameras, images, tracks, TRIANGULATOR_OPTIONS['filter_min_tri_angle'])
    num_observations_after = count_observations(tracks)
    return num_observations_before - num_observations_after

def complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS):
    num_completed_observations = complete_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS)
    print('Number of completed observations:', num_completed_observations)
    num_merged_observations = 0
    # TODO: Implement a better merge later, current version of merge does not have a good result and is not used in the pipeline
    # num_merged_observations = merge_tracks(cameras, images, tracks, TRIANGULATOR_OPTIONS)
    # print('Number of merged observations:', num_merged_observations)
    return num_completed_observations + num_merged_observations

def RetriangulateTracks(view_graph, cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS, BUNDLE_ADJUSTER_OPTIONS):
    # record status of images
    image_registered = images.is_registered.copy()
    pair_retry_counts = {}

    complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS)
    num_retriangulated_observations = retriangulate_underreconstructed_pairs(
        view_graph, cameras, images, tracks, TRIANGULATOR_OPTIONS, pair_retry_counts
    )
    print('Number of retriangulated observations:', num_retriangulated_observations)

    for i in range(TRIANGULATOR_OPTIONS['ba_global_max_refinements']):
        print(f'Running bundle adjustment iteration {i+1} / {TRIANGULATOR_OPTIONS["ba_global_max_refinements"]}') 
        num_observations_before = count_observations(tracks)
        ba_engine = TorchBA()
        ba_engine.Solve(cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS)
        num_changed_observations = 0
        num_changed_observations += abs(complete_and_merge_tracks(cameras, images, tracks, tracks_orig, TRIANGULATOR_OPTIONS))
        num_changed_observations += retriangulate_underreconstructed_pairs(
            view_graph, cameras, images, tracks, TRIANGULATOR_OPTIONS, pair_retry_counts
        )
        num_changed_observations += filter_points(
            cameras,
            images,
            tracks,
            TRIANGULATOR_OPTIONS,
            use_triangulation_angle=True,
        )
        changed_percentage = (
            num_changed_observations / num_observations_before
            if num_observations_before > 0
            else 0.0
        )
        if changed_percentage < TRIANGULATOR_OPTIONS['ba_global_max_refinement_change']:
            break
    
    # restore status of images
    images.is_registered[:] = image_registered
    return tracks
