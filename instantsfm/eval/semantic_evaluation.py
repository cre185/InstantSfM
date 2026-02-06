"""Semantic-aware reconstruction evaluation metrics.

Provides tools for evaluating per-object reconstruction quality and
analyzing semantic consistency in the reconstructed 3D points.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from instantsfm.scene.defs import Tracks, Images


def EvaluatePerObjectCompleteness(
    tracks: Tracks,
    ground_truth_points: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    distance_threshold: float = 0.1,
    verbose: bool = True
) -> Dict:
    """Evaluate reconstruction completeness per semantic object/instance.
    
    Compares reconstructed points against ground truth to measure per-object
    coverage and accuracy.
    
    Args:
        tracks: Tracks container with semantic_labels
        ground_truth_points: (M, 3) ground truth 3D points, or None
        ground_truth_labels: (M,) semantic labels for ground truth points, or None
        distance_threshold: Distance threshold for matching points (meters)
        verbose: If True, print detailed statistics
        
    Returns:
        Dictionary with per-object metrics:
        - 'per_object_stats': {label_id: {'precision', 'recall', 'f1', 'count'}}
        - 'overall_precision': Overall precision across all objects
        - 'overall_recall': Overall recall across all objects
    """
    stats = {
        'per_object_stats': {},
        'overall_precision': 0.0,
        'overall_recall': 0.0,
        'total_labeled_tracks': 0,
        'total_unlabeled_tracks': 0
    }
    
    # Count track labels
    labeled_mask = tracks.semantic_labels > 0
    stats['total_labeled_tracks'] = int(np.sum(labeled_mask))
    stats['total_unlabeled_tracks'] = len(tracks) - stats['total_labeled_tracks']
    
    if ground_truth_points is None or ground_truth_labels is None:
        if verbose:
            print("No ground truth provided, returning basic statistics only.")
        return stats
    
    # Group reconstructed points by label
    recon_by_label = defaultdict(list)
    for track_idx in range(len(tracks)):
        label = tracks.semantic_labels[track_idx]
        if label > 0:
            recon_by_label[label].append(tracks.xyzs[track_idx])
    
    # Group ground truth by label
    gt_by_label = defaultdict(list)
    for pt_idx, label in enumerate(ground_truth_labels):
        if label > 0:
            gt_by_label[label].append(ground_truth_points[pt_idx])
    
    # Evaluate per object
    all_labels = set(recon_by_label.keys()) | set(gt_by_label.keys())
    
    for label in all_labels:
        recon_pts = np.array(recon_by_label.get(label, []))
        gt_pts = np.array(gt_by_label.get(label, []))
        
        if len(recon_pts) == 0 or len(gt_pts) == 0:
            stats['per_object_stats'][int(label)] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'recon_count': len(recon_pts),
                'gt_count': len(gt_pts)
            }
            continue
        
        # Compute nearest neighbor distances
        from scipy.spatial import cKDTree
        
        recon_tree = cKDTree(recon_pts)
        gt_tree = cKDTree(gt_pts)
        
        # Precision: fraction of reconstructed points near ground truth
        dists_to_gt, _ = gt_tree.query(recon_pts)
        precision = np.sum(dists_to_gt < distance_threshold) / len(recon_pts)
        
        # Recall: fraction of ground truth points near reconstruction
        dists_to_recon, _ = recon_tree.query(gt_pts)
        recall = np.sum(dists_to_recon < distance_threshold) / len(gt_pts)
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        stats['per_object_stats'][int(label)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'recon_count': int(len(recon_pts)),
            'gt_count': int(len(gt_pts))
        }
    
    # Overall metrics (weighted by object size)
    total_recon = sum(s['recon_count'] for s in stats['per_object_stats'].values())
    total_gt = sum(s['gt_count'] for s in stats['per_object_stats'].values())
    
    if total_recon > 0:
        weighted_precision = sum(
            s['precision'] * s['recon_count'] 
            for s in stats['per_object_stats'].values()
        ) / total_recon
        stats['overall_precision'] = float(weighted_precision)
    
    if total_gt > 0:
        weighted_recall = sum(
            s['recall'] * s['gt_count'] 
            for s in stats['per_object_stats'].values()
        ) / total_gt
        stats['overall_recall'] = float(weighted_recall)
    
    if verbose:
        print(f"\n=== Per-Object Reconstruction Evaluation ===")
        print(f"Overall Precision: {stats['overall_precision']:.3f}")
        print(f"Overall Recall: {stats['overall_recall']:.3f}")
        print(f"\nPer-object results:")
        for label, obj_stats in sorted(stats['per_object_stats'].items()):
            print(f"  Instance {label}: P={obj_stats['precision']:.3f}, R={obj_stats['recall']:.3f}, "
                  f"F1={obj_stats['f1']:.3f} (recon={obj_stats['recon_count']}, gt={obj_stats['gt_count']})")
        print("=" * 50)
    
    return stats


def AnalyzeSemanticCrossContamination(
    tracks: Tracks,
    images: Images,
    verbose: bool = True
) -> Dict:
    """Analyze cross-instance contamination in tracks.
    
    Detects tracks that have observations spanning multiple semantic instances,
    indicating potential mismatches or boundary ambiguities.
    
    Args:
        tracks: Tracks container with semantic labels and observations
        images: Images container with per-image semantics
        verbose: If True, print analysis results
        
    Returns:
        Dictionary with contamination statistics:
        - 'pure_tracks': Tracks with single dominant label (>90% agreement)
        - 'mixed_tracks': Tracks with significant label mixing
        - 'contamination_ratio': Fraction of mixed tracks
    """
    from instantsfm.utils.semantic_utils import sample_label_at_points
    
    stats = {
        'pure_tracks': 0,
        'mixed_tracks': 0,
        'contamination_ratio': 0.0,
        'pure_track_ids': [],
        'mixed_track_ids': []
    }
    
    for track_idx in range(len(tracks)):
        track_label = tracks.semantic_labels[track_idx]
        
        if track_label <= 0:
            continue  # Skip unlabeled tracks
        
        observations = tracks.observations[track_idx]
        if len(observations) == 0:
            continue
        
        # Sample labels at all observations
        obs_labels = []
        for img_id, feat_id in observations:
            img_id, feat_id = int(img_id), int(feat_id)
            
            if img_id >= len(images) or images.semantics[img_id] is None:
                continue
            
            if feat_id >= len(images.features[img_id]):
                continue
            
            feat_coord = images.features[img_id][feat_id]
            label = sample_label_at_points(
                images.semantics[img_id],
                feat_coord.reshape(1, 2),
                neighborhood_size=3,
                mode='majority'
            )[0]
            
            if label > 0:
                obs_labels.append(label)
        
        if len(obs_labels) == 0:
            continue
        
        # Check label purity
        agreement = np.sum(np.array(obs_labels) == track_label) / len(obs_labels)
        
        if agreement >= 0.9:
            stats['pure_tracks'] += 1
            stats['pure_track_ids'].append(track_idx)
        else:
            stats['mixed_tracks'] += 1
            stats['mixed_track_ids'].append(track_idx)
    
    total_evaluated = stats['pure_tracks'] + stats['mixed_tracks']
    if total_evaluated > 0:
        stats['contamination_ratio'] = stats['mixed_tracks'] / total_evaluated
    
    if verbose:
        print(f"\n=== Semantic Cross-Contamination Analysis ===")
        print(f"Pure tracks (>90% agreement): {stats['pure_tracks']}")
        print(f"Mixed tracks: {stats['mixed_tracks']}")
        print(f"Contamination ratio: {stats['contamination_ratio']:.3f}")
        print("=" * 50)
    
    return stats


def ExportPerObjectPointClouds(
    tracks: Tracks,
    output_dir: str,
    min_points_per_object: int = 100,
    verbose: bool = True
) -> int:
    """Export per-object point clouds as separate PLY files.
    
    Args:
        tracks: Tracks container with semantic labels
        output_dir: Directory to save PLY files
        min_points_per_object: Minimum points required to export an object
        verbose: If True, print export statistics
        
    Returns:
        Number of object point clouds exported
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by label
    object_points = defaultdict(lambda: {'xyz': [], 'colors': []})
    
    for track_idx in range(len(tracks)):
        label = tracks.semantic_labels[track_idx]
        if label > 0:
            object_points[label]['xyz'].append(tracks.xyzs[track_idx])
            object_points[label]['colors'].append(tracks.colors[track_idx])
    
    exported_count = 0
    
    for label, data in object_points.items():
        if len(data['xyz']) < min_points_per_object:
            continue
        
        xyz = np.array(data['xyz'])
        colors = np.array(data['colors'])
        
        # Export as PLY
        ply_path = os.path.join(output_dir, f"object_{label:04d}.ply")
        _write_ply(ply_path, xyz, colors)
        exported_count += 1
    
    if verbose:
        print(f"Exported {exported_count} per-object point clouds to {output_dir}")
    
    return exported_count


def _write_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Write point cloud to PLY format."""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, color in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
