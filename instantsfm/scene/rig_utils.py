import numpy as np
import pypose as pp
from scipy.spatial.transform import Rotation as R
from collections import defaultdict


# ============================================================================
# Rig Conversion Functions
# ============================================================================

def Single2Rig(images, cameras=None, tracks=None, VOTING_OPTIONS=None, use_fixed_rel_poses=False):
    """
    Convert single camera poses to rig structure (ref_poses and rel_poses).
    Automatically handles voting if needed.
    
    Args:
        images: Images object with world2cams in single format
        cameras: Camera objects (needed for voting)
        tracks: Track objects (needed for voting)
        VOTING_OPTIONS: Configuration for voting (used if needed)
        use_fixed_rel_poses: Whether to use fixed relative poses from file
    
    Returns:
        None (modifies images in-place)
    """    
    if use_fixed_rel_poses:
        print('Using fixed relative camera poses')
        _build_rig_from_fixed_poses(images)
    else:
        if images.ref_camera_idx is None:
            if cameras is None or tracks is None or VOTING_OPTIONS is None:
                raise ValueError("Need cameras, tracks, and VOTING_OPTIONS for voting")
            print('Auto-invoking reference camera voting')
            _calculate_votes(cameras, images, tracks, VOTING_OPTIONS)
        _build_rig_from_voting(images, VOTING_OPTIONS)

def Rig2Single(images):
    """
    Reconstruct single camera world2cams from rig structure (ref_poses and rel_poses).
    Args:
        images: Images object with ref_poses and rel_poses
    
    Returns:
        None (modifies images.world2cams in-place)
    """
    # Extract rig indices for valid images
    group_idx = images.image_to_rig[:, 0]
    member_idx = images.image_to_rig[:, 1]
    images.world2cams = images.rel_poses[member_idx] @ images.ref_poses[group_idx]


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _calculate_votes(cameras, images, tracks, VOTING_OPTIONS):
    # Count matches for each image
    image_match_counts = _count_matches_per_image(images, tracks)
    
    # Compute and store image votes based on ranking within each group
    num_images = len(images)
    num_groups = images.rig_groups.shape[0]
    num_columns = images.rig_groups.shape[1]
    image_votes = np.zeros(num_images)
    
    # First, vote for reference camera to know which column to check    
    num_groups = images.rig_groups.shape[0]
    num_columns = images.rig_groups.shape[1]
    
    group_matches = np.zeros(num_groups)
    column_votes = np.zeros(num_columns)
    
    for group_idx in range(num_groups):
        img_ids = images.rig_groups[group_idx]
        
        valid_indices = []
        valid_img_ids = []
        for cidx, img_id in enumerate(img_ids):
            if img_id != -1 and images.is_registered[img_id]:
                valid_indices.append(cidx)
                valid_img_ids.append(img_id)
        
        if len(valid_img_ids) < 2:
            continue
        
        match_counts = [image_match_counts.get(img_id, 0) for img_id in valid_img_ids]
        group_matches[group_idx] = sum(match_counts)
        ranked_indices = np.argsort(match_counts)[::-1]
        votes = np.arange(len(valid_img_ids), 0, -1)
        
        for rank, idx in enumerate(ranked_indices):
            img_id = valid_img_ids[idx]
            image_votes[img_id] = votes[rank]
            cidx = valid_indices[idx]
            column_votes[cidx] += votes[rank]
    
    ref_camera_idx = int(np.argmax(column_votes))
    
    # Compute group-level weights based on total matches per group
    group_matches = sorted(enumerate(group_matches), key=lambda x: x[1], reverse=True)
    
    # Assign group weights: linearly interpolate from 3.0 (best) to 1.0 (worst)
    group_weights = np.ones(num_groups)
    valid_groups = [gidx for gidx, votes in group_matches if votes > 0]
    num_valid_groups = len(valid_groups)

    for rank, (group_idx, _) in enumerate(group_matches):
        if group_idx in valid_groups:
            # Linear interpolation: rank 0 -> weight 3.0, rank (num_valid_groups-1) -> weight 1.0
            weight = 3.0 - (2.0 * rank / (num_valid_groups - 1))
            group_weights[group_idx] = weight
    
    # Apply group weights to image votes
    for group_idx in range(num_groups):
        img_ids = images.rig_groups[group_idx]
        group_weight = group_weights[group_idx]
        for img_id in img_ids:
            image_votes[img_id] *= group_weight
    
    # Store results in images object
    images.image_votes = image_votes
    images.ref_camera_idx = ref_camera_idx
    ref_folder_name = images.rig_folder_names[ref_camera_idx]
    print(f'Reference camera selected: {ref_folder_name} (column {ref_camera_idx})')


def _count_matches_per_image(images, tracks):
    """Count the number of track observations for each image."""
    match_counts = defaultdict(int)
    
    for track_id in range(len(tracks)):
        observations = tracks.observations[track_id]
        for img_id, _ in observations:
            if images.is_registered[img_id]:
                match_counts[img_id] += 1
    
    return dict(match_counts)


def _build_rig_from_fixed_poses(images):
    """Build rig structure using fixed relative poses."""
    num_groups = images.rig_groups.shape[0]
    num_columns = images.rig_groups.shape[1]
    
    ref_poses = np.tile(np.eye(4), (num_groups, 1, 1))
    
    # Use pre-aligned fixed_rel_poses (already in correct order and format)
    rel_poses = images.fixed_rel_poses.copy()
    ref_camera_idx = images.ref_camera_idx
    
    # Set reference poses from registered images
    for group_idx in range(num_groups):
        img_ids = images.rig_groups[group_idx]
        ref_img_id = img_ids[ref_camera_idx]
        
        if ref_img_id != -1 and images.is_registered[ref_img_id]:
            ref_poses[group_idx] = images.world2cams[ref_img_id].copy()
    
    images.ref_poses = ref_poses
    images.rel_poses = rel_poses


def _build_rig_from_voting(images, VOTING_OPTIONS):
    """Build rig structure using voting results."""
    num_groups = images.rig_groups.shape[0]
    num_columns = images.rig_groups.shape[1]
    ref_camera_idx = images.ref_camera_idx
    
    ref_poses = np.tile(np.eye(4), (num_groups, 1, 1))
    rel_poses = np.tile(np.eye(4), (num_columns, 1, 1))
    
    rel_rotations = [[] for _ in range(num_columns)]
    rel_translations = [[] for _ in range(num_columns)]
    rel_votes = [[] for _ in range(num_columns)]
    
    for group_idx in range(num_groups):
        img_ids = images.rig_groups[group_idx]
        ref_img_id = img_ids[ref_camera_idx]
        
        if not images.is_registered[ref_img_id]:
            continue
        
        ref_pose = pp.mat2SE3(images.world2cams[ref_img_id])
        for cidx, img_id in enumerate(img_ids):
            if cidx == ref_camera_idx:
                continue
            
            img_pose = pp.mat2SE3(images.world2cams[img_id])
            rel_pose = img_pose * ref_pose.Inv()
            rel_rotation = rel_pose.rotation().cpu().numpy()
            rel_rotations[cidx].append(rel_rotation)
            rel_translation = rel_pose.translation().cpu().numpy()
            rel_translations[cidx].append(rel_translation)
            rel_votes[cidx].append(images.image_votes[img_id])
    
    for cidx in range(num_columns):
        if cidx == ref_camera_idx:
            rel_poses[cidx] = np.eye(4)
        else:
            rotations = np.array(rel_rotations[cidx])
            translations = np.array(rel_translations[cidx])
            votes = np.array(rel_votes[cidx])
            best_rotation, _ = _form_parties_and_vote_rotation(rotations, votes, VOTING_OPTIONS)
            best_translation, _ = _form_parties_and_vote_translation(translations, votes, VOTING_OPTIONS)
            rel_poses[cidx, :3, :3] = R.from_quat(best_rotation).as_matrix()
            rel_poses[cidx, :3, 3] = best_translation
    
    for group_idx in range(num_groups):
        img_ids = images.rig_groups[group_idx]
        ref_img_id = img_ids[ref_camera_idx]
        
        if images.is_registered[ref_img_id]:
            ref_poses[group_idx] = images.world2cams[ref_img_id].copy()
    
    images.ref_poses = ref_poses
    images.rel_poses = rel_poses


def _form_parties_and_vote_translation(translations, votes, VOTING_OPTIONS):
    """
    Form parties of similar translations and vote for the best one.
    
    Args:
        translations: Array of translation vectors (N, 3)
        votes: Array of votes (N,)
        VOTING_OPTIONS: Configuration options
    
    Returns:
        best_translation: The winning translation vector
        party_info: Dict with info about the winning party
    """
    n_translations = len(translations)
    
    # Adaptive threshold adjustment
    distance_threshold = VOTING_OPTIONS['initial_distance_threshold']
    target_ratio = VOTING_OPTIONS['target_party_ratio']
    distance_step = VOTING_OPTIONS['distance_threshold_step']
    min_threshold = VOTING_OPTIONS['min_distance_threshold']
    max_threshold = VOTING_OPTIONS['max_distance_threshold']
    
    best_party = None
    best_threshold = distance_threshold
    
    # Try different thresholds to find one that gives target_ratio
    for iteration in range(20):
        parties = []
        assigned = np.zeros(n_translations, dtype=bool)
        
        # Form parties greedily
        while not np.all(assigned):
            # Find unassigned translation with highest vote
            unassigned_indices = np.where(~assigned)[0]
            seed_idx = unassigned_indices[np.argmax(votes[unassigned_indices])]
            
            # Form party around this seed
            party_members = [seed_idx]
            party_votes = [votes[seed_idx]]
            seed_translation = translations[seed_idx]
            
            # Add similar translations to this party
            for idx in unassigned_indices:
                if idx == seed_idx:
                    continue
                
                dist = np.linalg.norm(translations[idx] - seed_translation)
                if dist <= distance_threshold:
                    party_members.append(idx)
                    party_votes.append(votes[idx])
            
            # Mark as assigned
            assigned[party_members] = True
            
            parties.append({
                'members': party_members,
                'votes': sum(party_votes),
                'seed_translation': seed_translation,
                'size': len(party_members)
            })
        
        # Find largest party
        largest_party = max(parties, key=lambda p: p['votes'])
        largest_ratio = largest_party['size'] / n_translations
        
        # Check if we've reached target ratio
        if abs(largest_ratio - target_ratio) < 0.1 or iteration > 15:
            best_party = largest_party
            best_threshold = distance_threshold
            break
        
        # Adjust threshold
        if largest_ratio < target_ratio:
            distance_threshold += distance_step
            if distance_threshold > max_threshold:
                distance_threshold = max_threshold
                best_party = largest_party
                best_threshold = distance_threshold
                break
        else:
            distance_threshold -= distance_step
            if distance_threshold < min_threshold:
                distance_threshold = min_threshold
                best_party = largest_party
                best_threshold = distance_threshold
                break
        
        best_party = largest_party
        best_threshold = distance_threshold
    
    # Weighted average of translations in party
    party_translations = translations[best_party['members']]
    party_weights = votes[best_party['members']]
    winning_translation = _weighted_translation_average(party_translations, party_weights)
    
    party_info = {
        'size': best_party['size'],
        'votes': best_party['votes'],
        'threshold': best_threshold,
        'ratio': best_party['size'] / n_translations
    }
    
    return winning_translation, party_info


def _quaternion_angle_distance(q1, q2):
    """
    Compute the angular distance between two quaternions in degrees.
    
    Args:
        q1, q2: Quaternions in [qx, qy, qz, qw] format
    
    Returns:
        Angular distance in degrees
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product (cosine of half angle)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Convert to angle in degrees
    angle = 2 * np.arccos(dot) * 180 / np.pi
    
    return angle


def _form_parties_and_vote_rotation(rotations, votes, VOTING_OPTIONS):
    """
    Form parties of similar rotations and vote for the best one.
    
    Args:
        rotations: Array of quaternions (N, 4) in [qx, qy, qz, qw] format
        votes: Array of votes (N,)
    
    Returns:
        best_rotation: The winning rotation
        party_info: Dict with info about the winning party
    """
    n_rotations = len(rotations)
    
    # Adaptive threshold adjustment
    angle_threshold = VOTING_OPTIONS['initial_angle_threshold']
    target_ratio = VOTING_OPTIONS['target_party_ratio']
    
    best_party = None
    best_threshold = angle_threshold
    
    # Try different thresholds to find one that gives target_ratio
    for iteration in range(10):
        parties = []
        assigned = np.zeros(n_rotations, dtype=bool)
        
        # Form parties greedily
        while not np.all(assigned):
            # Find unassigned rotation with highest vote
            unassigned_indices = np.where(~assigned)[0]
            seed_idx = unassigned_indices[np.argmax(votes[unassigned_indices])]
            
            # Form party around this seed
            party_members = [seed_idx]
            party_votes = [votes[seed_idx]]
            seed_rotation = rotations[seed_idx]
            
            # Add similar rotations to this party
            for idx in unassigned_indices:
                if idx == seed_idx:
                    continue
                
                angle_dist = _quaternion_angle_distance(rotations[idx], seed_rotation)
                if angle_dist <= angle_threshold:
                    party_members.append(idx)
                    party_votes.append(votes[idx])
            
            # Mark as assigned
            assigned[party_members] = True
            
            parties.append({
                'members': party_members,
                'votes': sum(party_votes),
                'seed_rotation': seed_rotation,
                'size': len(party_members)
            })
        
        # Find largest party
        largest_party = max(parties, key=lambda p: p['votes'])
        largest_ratio = largest_party['size'] / n_rotations
        
        # Check if we've reached target ratio
        if abs(largest_ratio - target_ratio) < 0.05:
            best_party = largest_party
            best_threshold = angle_threshold
            break
        
        # Adjust threshold
        if largest_ratio < target_ratio:
            angle_threshold += VOTING_OPTIONS['angle_threshold_step']
            if angle_threshold > VOTING_OPTIONS['max_angle_threshold']:
                angle_threshold = VOTING_OPTIONS['max_angle_threshold']
                best_party = largest_party
                best_threshold = angle_threshold
                break
        else:
            angle_threshold -= VOTING_OPTIONS['angle_threshold_step']
            if angle_threshold < VOTING_OPTIONS['min_angle_threshold']:
                angle_threshold = VOTING_OPTIONS['min_angle_threshold']
                best_party = largest_party
                best_threshold = angle_threshold
                break
        
        best_party = largest_party
        best_threshold = angle_threshold
    
    # Average quaternions in party
    party_rotations = rotations[best_party['members']]
    party_weights = votes[best_party['members']]
    winning_rotation = _weighted_quaternion_average(party_rotations, party_weights)
    
    party_info = {
        'size': best_party['size'],
        'votes': best_party['votes'],
        'threshold': best_threshold,
        'ratio': best_party['size'] / n_rotations
    }
    
    return winning_rotation, party_info


def _weighted_translation_average(translations, weights):
    """Compute weighted average of translation vectors."""
    weights = weights / np.sum(weights)
    avg_translation = np.sum(translations * weights[:, np.newaxis], axis=0)
    return avg_translation


def _weighted_quaternion_average(quaternions, weights):
    """Compute weighted average of quaternions using Markley's method."""
    weights = weights / np.sum(weights)
    
    M = np.zeros((4, 4))
    for i, q in enumerate(quaternions):
        q = q / np.linalg.norm(q)
        q_wxyz = np.array([q[3], q[0], q[1], q[2]])
        M += weights[i] * np.outer(q_wxyz, q_wxyz)
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_quat_wxyz = eigenvectors[:, -1]
    
    avg_quat = np.array([avg_quat_wxyz[1], avg_quat_wxyz[2], avg_quat_wxyz[3], avg_quat_wxyz[0]])
    
    if avg_quat[3] < 0:
        avg_quat = -avg_quat
    
    return avg_quat
