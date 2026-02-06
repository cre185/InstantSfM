"""Semantic segmentation utilities for instance-aware SfM.

This module provides tools for:
- Sampling semantic/instance labels at feature points
- Filtering matches based on semantic consistency
- Aggregating per-track semantic labels

Semantic format: Per-pixel instance IDs as (H, W) int arrays.
- Background/unknown: 0 or -1
- Instance IDs: positive integers (1, 2, 3, ...)
"""

import numpy as np
from typing import Optional, Tuple
import cv2


def sample_label_at_points(
    sem_mask: Optional[np.ndarray],
    points: np.ndarray,
    neighborhood_size: int = 3,
    mode: str = 'majority'
) -> np.ndarray:
    """Sample semantic/instance labels at feature point locations.
    
    Args:
        sem_mask: (H, W) array of instance IDs, or None
        points: (N, 2) array of (x, y) pixel coordinates
        neighborhood_size: Size of sampling neighborhood (odd integer)
        mode: 'majority' (most common label) or 'center' (nearest pixel)
        
    Returns:
        (N,) array of instance labels. Returns -1 for all if sem_mask is None.
    """
    if sem_mask is None or len(points) == 0:
        return np.full(len(points), -1, dtype=np.int32)
    
    h, w = sem_mask.shape
    labels = np.full(len(points), -1, dtype=np.int32)
    
    if mode == 'center':
        # Simple nearest-neighbor sampling
        x_int = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
        y_int = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)
        labels = sem_mask[y_int, x_int]
    
    elif mode == 'majority':
        # Majority voting in a small neighborhood
        half_size = neighborhood_size // 2
        
        for i, (x, y) in enumerate(points):
            x_int, y_int = int(round(x)), int(round(y))
            
            # Extract neighborhood with boundary checks
            y_min = max(0, y_int - half_size)
            y_max = min(h, y_int + half_size + 1)
            x_min = max(0, x_int - half_size)
            x_max = min(w, x_int + half_size + 1)
            
            if y_max > y_min and x_max > x_min:
                patch = sem_mask[y_min:y_max, x_min:x_max]
                # Get most common non-zero label
                valid_labels = patch[patch > 0]
                if len(valid_labels) > 0:
                    labels[i] = np.bincount(valid_labels).argmax()
                else:
                    labels[i] = 0  # Background
    
    return labels


def consistent_match_mask(
    matches: np.ndarray,
    feats1: np.ndarray,
    feats2: np.ndarray,
    sem1: Optional[np.ndarray],
    sem2: Optional[np.ndarray],
    neighborhood_size: int = 3,
    min_agreement: float = 0.6
) -> np.ndarray:
    """Filter matches to keep only semantically consistent pairs.
    
    A match is considered consistent if:
    1. Both feature points have valid semantic labels (not background/unknown)
    2. The labels from both images agree
    
    Args:
        matches: (M, 2) array of (idx1, idx2) match indices
        feats1: (N1, 2) array of feature coordinates in image 1
        feats2: (N2, 2) array of feature coordinates in image 2
        sem1: (H1, W1) semantic mask for image 1, or None
        sem2: (H2, W2) semantic mask for image 2, or None
        neighborhood_size: Size of sampling neighborhood for label extraction
        min_agreement: Minimum fraction of neighborhood that must agree (0-1)
        
    Returns:
        (M,) boolean mask - True for consistent matches
    """
    if sem1 is None or sem2 is None or len(matches) == 0:
        # If semantics unavailable, keep all matches (no filtering)
        return np.ones(len(matches), dtype=bool)
    
    # Extract matched feature coordinates
    idx1, idx2 = matches[:, 0], matches[:, 1]
    
    # Bounds checking
    valid_idx1 = (idx1 >= 0) & (idx1 < len(feats1))
    valid_idx2 = (idx2 >= 0) & (idx2 < len(feats2))
    valid_base = valid_idx1 & valid_idx2
    
    if not np.any(valid_base):
        return np.zeros(len(matches), dtype=bool)
    
    # Sample labels at matched feature locations
    matched_feats1 = feats1[idx1[valid_base]]
    matched_feats2 = feats2[idx2[valid_base]]
    
    labels1 = sample_label_at_points(sem1, matched_feats1, neighborhood_size, mode='majority')
    labels2 = sample_label_at_points(sem2, matched_feats2, neighborhood_size, mode='majority')
    
    # Check consistency: labels match and are valid (non-background)
    consistent_valid = (labels1 == labels2) & (labels1 > 0) & (labels2 > 0)
    
    # For background or unknown labels, keep the match (neutral stance)
    background_or_unknown = (labels1 <= 0) | (labels2 <= 0)
    
    # Final mask: consistent OR background (don't penalize uncertain regions)
    consistent = consistent_valid | background_or_unknown
    
    # Map back to full match array
    full_mask = np.zeros(len(matches), dtype=bool)
    full_mask[valid_base] = consistent
    
    return full_mask


def aggregate_track_semantics(
    observations: np.ndarray,
    images_semantics: list,
    images_features: list,
    neighborhood_size: int = 3,
    min_confidence: float = 0.5
) -> Tuple[int, float]:
    """Aggregate semantic labels for a single track from its observations.
    
    Args:
        observations: (N, 2) array of (image_id, feature_id) pairs
        images_semantics: List of per-image semantic masks (may contain None)
        images_features: List of per-image feature arrays
        neighborhood_size: Size of sampling neighborhood
        min_confidence: Minimum confidence to assign a label (else return -1)
        
    Returns:
        (dominant_label, confidence) tuple
        - dominant_label: Most common instance ID, or -1 if insufficient confidence
        - confidence: Proportion of observations agreeing with dominant label
    """
    if len(observations) == 0:
        return -1, 0.0
    
    label_votes = []
    
    for img_id, feat_id in observations:
        img_id, feat_id = int(img_id), int(feat_id)
        
        # Check bounds
        if img_id < 0 or img_id >= len(images_semantics):
            continue
        if img_id >= len(images_features) or feat_id >= len(images_features[img_id]):
            continue
        
        sem_mask = images_semantics[img_id]
        if sem_mask is None:
            continue
        
        feat_coord = images_features[img_id][feat_id]
        label = sample_label_at_points(sem_mask, feat_coord.reshape(1, 2), neighborhood_size, mode='majority')[0]
        
        if label > 0:  # Valid instance label
            label_votes.append(label)
    
    if len(label_votes) == 0:
        return -1, 0.0
    
    # Count votes
    unique_labels, counts = np.unique(label_votes, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_label = int(unique_labels[dominant_idx])
    confidence = float(counts[dominant_idx]) / len(label_votes)
    
    # Return label only if confidence meets threshold
    if confidence >= min_confidence:
        return dominant_label, confidence
    else:
        return -1, confidence


def compute_semantic_consistency_score(
    labels1: np.ndarray,
    labels2: np.ndarray,
    ignore_background: bool = True
) -> float:
    """Compute consistency score between two sets of semantic labels.
    
    Args:
        labels1: (N,) array of instance labels
        labels2: (N,) array of instance labels
        ignore_background: If True, ignore comparisons where either label is <= 0
        
    Returns:
        Consistency score in [0, 1] - proportion of matching labels
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0
    
    if ignore_background:
        valid_mask = (labels1 > 0) & (labels2 > 0)
        if not np.any(valid_mask):
            return 1.0  # No valid comparisons, assume consistent
        return float(np.sum(labels1[valid_mask] == labels2[valid_mask])) / np.sum(valid_mask)
    else:
        return float(np.sum(labels1 == labels2)) / len(labels1)
