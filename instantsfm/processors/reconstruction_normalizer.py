import numpy as np

def NormalizeReconstruction(images, tracks, use_depths=False, fixed_scale=False, extent=10., p0=0.1, p1=0.9):
    # Batch compute image centers
    Rs = images.world2cams[:, :3, :3]  # (N, 3, 3)
    ts = images.world2cams[:, :3, 3]   # (N, 3)
    coords = -np.einsum('nij,nj->ni', Rs.transpose(0, 2, 1), ts)  # (N, 3)
    coords_sorted = np.sort(coords, axis=0)
    P0 = int(p0 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else 0
    P1 = int(p1 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else coords.shape[0] - 1
    bbox_min = coords_sorted[P0]
    bbox_max = coords_sorted[P1]
    mean_coord = np.mean(coords_sorted[P0:P1+1], axis=0)

    if use_depths:
        # depth-based normalization
        depth_gt_list = []
        depth_pred_list = []
        for track_id in range(len(tracks)):
            for image_id, feature_id in tracks.observations[track_id]:
                depth_gt = images.depths[image_id][feature_id]
                if depth_gt > 0:
                    C = coords[image_id]  # Already computed
                    P = tracks.xyzs[track_id]
                    depth_pred = np.linalg.norm(P - C)
                    depth_gt_list.append(depth_gt)
                    depth_pred_list.append(depth_pred)
        if len(depth_gt_list) > 0:
            log_scales = np.log(np.array(depth_gt_list)) - np.log(np.array(depth_pred_list))
            scale = np.exp(np.median(log_scales))
        else:
            scale = 1.0
    else:
        # default normalization
        scale = 1.
        if not fixed_scale:
            old_extent = np.linalg.norm(bbox_max - bbox_min)
            if old_extent >= 1e-6:
                scale = extent / old_extent
        
    
    coords = (coords - mean_coord) * scale
    # Batch update translations
    images.world2cams[:, :3, 3] = -np.einsum('nij,nj->ni', images.world2cams[:, :3, :3], coords)
    # Batch update track positions
    tracks.xyzs[:] = (tracks.xyzs - mean_coord) * scale
