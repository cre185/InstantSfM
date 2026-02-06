"""Semantic filtering processor for dynamic object detection.

This module provides semantic-based filtering in two stages:
1. Relative pose stage: Per-camera dynamic object detection
2. Track stage: Cross-camera object association and filtering
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from instantsfm.scene.defs import Images, ViewGraph, Objects, Object, Tracks
from instantsfm.utils.semantic_utils import sample_label_at_points
from instantsfm.utils.union_find import UnionFind


class SemanticFilter:
    """Semantic filtering processor for multi-stage dynamic object detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_observations = config.get('min_observations_for_dynamic', 3)
        self.dynamic_threshold = config.get('dynamic_inlier_threshold', 0.3)
        self.use_percentile = config.get('use_percentile', 25)
        self.verbose = config.get('verbose_semantics', False)
    
    def detect_dynamic_per_camera(
        self,
        view_graph: ViewGraph,
        images: Images
    ) -> Dict:
        """Detect dynamic objects within each camera separately.
        
        Only analyzes temporal sequences within the same camera.
        
        Returns:
            Dict mapping camera_key -> set of dynamic label_ids
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Detecting Dynamic Objects Per-Camera")
            print("="*60)
        
        # Group images by camera folder (for multi-rig) or by camera_id (for single)
        camera_groups = defaultdict(list)
        
        if images.rig_groups.size > 0:
            # Multi-camera rig: group by folder (camera position)
            for img_id in range(len(images)):
                group_idx, cam_idx = images.image_to_rig[img_id]
                if cam_idx >= 0:
                    folder_name = images.rig_folder_names[cam_idx]
                    camera_groups[folder_name].append(img_id)
        else:
            # Single camera or no rig: group by camera_id
            for img_id in range(len(images)):
                cam_id = images.cam_ids[img_id]
                camera_groups[cam_id].append(img_id)
        
        # For each camera, collect inlier statistics for each label
        camera_dynamic_labels = {}
        
        for camera_key, img_ids in camera_groups.items():
            if self.verbose:
                print(f"\nAnalyzing camera: {camera_key} ({len(img_ids)} images)")
            
            # Collect label inlier statistics within this camera
            label_stats = defaultdict(list)
            pairs_processed = 0
            
            img_id_set = set(img_ids)
            
            for pair in view_graph.image_pairs.values():
                img_i, img_j = pair.image_id1, pair.image_id2
                
                # Only process pairs within the same camera
                if img_i not in img_id_set or img_j not in img_id_set:
                    continue
                
                # Skip if either image has no semantics
                if images.semantics[img_i] is None or images.semantics[img_j] is None:
                    continue
                
                if len(pair.inliers) == 0 or len(pair.matches) == 0:
                    continue
                
                pairs_processed += 1
                
                # Get all matches and inliers
                all_matches = pair.matches
                inlier_matches = pair.matches[pair.inliers]
                
                # Sample semantic labels
                all_feats_i = images.features[img_i][all_matches[:, 0]]
                inlier_feats_i = images.features[img_i][inlier_matches[:, 0]]
                
                all_labels_i = sample_label_at_points(images.semantics[img_i], all_feats_i, mode='center')
                inlier_labels_i = sample_label_at_points(images.semantics[img_i], inlier_feats_i, mode='center')
                
                # For each label, compute inlier ratio
                unique_labels = np.unique(all_labels_i)
                for label in unique_labels:
                    if label == 0:  # Skip background
                        continue
                    
                    num_total = np.sum(all_labels_i == label)
                    num_inliers = np.sum(inlier_labels_i == label)
                    
                    if num_total > 0:
                        inlier_ratio = num_inliers / num_total
                        label_stats[label].append(inlier_ratio)
            
            # Analyze and mark dynamic labels
            dynamic_labels = set()
            for label, ratios in label_stats.items():
                if len(ratios) < self.min_observations:
                    continue
                
                percentile_ratio = np.percentile(ratios, self.use_percentile)
                if percentile_ratio < self.dynamic_threshold:
                    dynamic_labels.add(label)
                    if self.verbose:
                        print(f"  Label {label}: p{self.use_percentile}={percentile_ratio:.3f} "
                              f"(n={len(ratios)}) -> DYNAMIC")
            
            camera_dynamic_labels[camera_key] = dynamic_labels
            
            if self.verbose:
                print(f"  Processed {pairs_processed} pairs, detected {len(dynamic_labels)} dynamic labels")
        
        if self.verbose:
            total_dynamic = sum(len(labels) for labels in camera_dynamic_labels.values())
            print(f"\n{'='*60}")
            print(f"Total: {len(camera_groups)} cameras, {total_dynamic} dynamic labels")
            print(f"{'='*60}")
        
        return camera_dynamic_labels
    
    def filter_matches_by_dynamic_labels(
        self,
        view_graph: ViewGraph,
        images: Images,
        camera_dynamic_labels: Dict
    ):
        """Filter matches on dynamic labels within each camera."""
        
        if self.verbose:
            print("\nFiltering matches on dynamic labels...")
        
        total_matches = 0
        total_filtered = 0
        pairs_processed = 0
        
        # Build image_id -> camera_key mapping
        img_to_camera = {}
        if images.rig_groups.size > 0:
            for img_id in range(len(images)):
                group_idx, cam_idx = images.image_to_rig[img_id]
                if cam_idx >= 0:
                    img_to_camera[img_id] = images.rig_folder_names[cam_idx]
        else:
            for img_id in range(len(images)):
                img_to_camera[img_id] = images.cam_ids[img_id]
        
        for pair in view_graph.image_pairs.values():
            img_i, img_j = pair.image_id1, pair.image_id2
            
            # Only filter within-camera pairs
            if img_to_camera.get(img_i) != img_to_camera.get(img_j):
                continue
            
            camera_key = img_to_camera.get(img_i)
            if camera_key is None or camera_key not in camera_dynamic_labels:
                continue
            
            dynamic_labels = camera_dynamic_labels[camera_key]
            if len(dynamic_labels) == 0:
                continue
            
            if images.semantics[img_i] is None or images.semantics[img_j] is None:
                continue
            
            if len(pair.inliers) == 0:
                continue
            
            pairs_processed += 1
            
            # Get inlier matches and their labels
            inlier_matches = pair.matches[pair.inliers]
            num_before = len(inlier_matches)
            total_matches += num_before
            
            feats_i = images.features[img_i][inlier_matches[:, 0]]
            labels_i = sample_label_at_points(images.semantics[img_i], feats_i, mode='center')
            
            # Keep inliers that are NOT on dynamic labels
            keep_mask = np.array([label not in dynamic_labels for label in labels_i])
            num_filtered = len(keep_mask) - np.sum(keep_mask)
            
            if num_filtered > 0:
                kept_indices = pair.inliers[keep_mask]
                pair.inliers = kept_indices
                total_filtered += num_filtered
        
        print(f'Filtered {total_filtered} / {total_matches} matches on dynamic objects ({100*total_filtered/max(total_matches,1):.1f}%)')
    
    def associate_objects_from_tracks(
        self,
        tracks: Tracks,
        images: Images
    ) -> Objects:
        """Associate objects across cameras using established tracks.
        
        Uses 3D track information to reliably associate objects.
        
        Returns:
            Objects container with cross-camera associations
        """
        min_observations = self.config.get('min_observations_for_object', 3)
        min_consistency = self.config.get('min_object_consistency', 0.7)
        
        if self.verbose:
            print("\n" + "="*60)
            print("Associating Objects from Tracks")
            print("="*60)
        
        # For each track, collect (image_id, label_id) pairs
        uf = UnionFind()
        
        for track_idx in range(len(tracks)):
            track = tracks[track_idx]
            obs = track.observations  # (N, 2): (image_id, feature_id)
            
            if len(obs) < min_observations:
                continue
            
            # Collect labels for this track
            labels_in_track = []
            for img_id, feat_idx in obs:
                if images.semantics[img_id] is None:
                    continue
                
                feat = images.features[img_id][feat_idx:feat_idx+1]
                label = sample_label_at_points(images.semantics[img_id], feat, mode='center')[0]
                
                if label > 0:  # Skip background
                    labels_in_track.append((img_id, label))
            
            if len(labels_in_track) < 2:
                continue
            
            # Check label consistency within this track
            label_ids = [label for _, label in labels_in_track]
            unique, counts = np.unique(label_ids, return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            consistency = counts[np.argmax(counts)] / len(label_ids)
            
            if consistency < min_consistency:
                continue
            
            # Union all (image, label) pairs in this track
            filtered_pairs = [(img, lbl) for img, lbl in labels_in_track if lbl == most_common_label]
            
            if len(filtered_pairs) >= 2:
                for i in range(len(filtered_pairs) - 1):
                    uf.Union(filtered_pairs[i], filtered_pairs[i+1])
        
        # Extract connected components -> global objects
        objects = Objects()
        component_map = {}
        
        all_nodes = set()
        for track_idx in range(len(tracks)):
            track = tracks[track_idx]
            obs = track.observations
            for img_id, feat_idx in obs:
                if images.semantics[img_id] is None:
                    continue
                feat = images.features[img_id][feat_idx:feat_idx+1]
                label = sample_label_at_points(images.semantics[img_id], feat, mode='center')[0]
                if label > 0:
                    node = (img_id, label)
                    all_nodes.add(node)
                    root = uf.Find(node)
                    
                    if root not in component_map:
                        obj = objects.create_object()
                        component_map[root] = obj.object_id
                    
                    object_id = component_map[root]
                    objects.add_observation(object_id, img_id, label)
        
        if self.verbose:
            print(f"Created {len(objects)} global objects from tracks")
            if len(objects) > 0:
                sizes = [len(obj.image_labels) for obj in objects.objects.values()]
                print(f"  Object visibility: {min(sizes)}-{max(sizes)} images (avg: {np.mean(sizes):.1f})")
        
        return objects
    
    def detect_dynamic_objects_from_tracks(
        self,
        tracks: Tracks,
        images: Images,
        objects: Objects
    ):
        """Detect dynamic objects by analyzing track reprojection statistics.
        
        Marks objects as dynamic based on track quality metrics.
        """
        if len(objects) == 0:
            return
        
        min_observations = self.config.get('min_observations_for_object', 3)
        dynamic_threshold = self.config.get('track_length_ratio_threshold', 0.5)
        
        if self.verbose:
            print("\nDetecting dynamic objects from track statistics...")
        
        # For each object, collect track statistics
        object_track_stats = {obj_id: {'track_lengths': [], 'track_indices': []} 
                             for obj_id in objects.objects.keys()}
        
        for track_idx in range(len(tracks)):
            track = tracks[track_idx]
            obs = track.observations
            
            if len(obs) < 3:
                continue
            
            # Find which object this track belongs to
            obj_ids = []
            for img_id, feat_idx in obs:
                if images.semantics[img_id] is None:
                    continue
                
                feat = images.features[img_id][feat_idx:feat_idx+1]
                label = sample_label_at_points(images.semantics[img_id], feat, mode='center')[0]
                obj_id = objects.get_object_id(img_id, label)
                
                if obj_id is not None:
                    obj_ids.append(obj_id)
            
            if len(obj_ids) == 0:
                continue
            
            # Find most common object for this track
            unique_ids, counts = np.unique(obj_ids, return_counts=True)
            main_obj_id = unique_ids[np.argmax(counts)]
            
            object_track_stats[main_obj_id]['track_lengths'].append(len(obs))
            object_track_stats[main_obj_id]['track_indices'].append(track_idx)
        
        # Analyze each object
        num_dynamic = 0
        for obj_id, stats in object_track_stats.items():
            if len(stats['track_lengths']) < min_observations:
                continue
            
            track_lengths = np.array(stats['track_lengths'])
            
            # Dynamic objects tend to have shorter tracks (fewer consistent observations)
            # Compare to median track length across all objects
            median_length = np.median(track_lengths)
            p25_length = np.percentile(track_lengths, 25)
            
            # Also check how many very short tracks this object has
            very_short_ratio = np.sum(track_lengths <= 3) / len(track_lengths)
            
            # Mark as dynamic if tracks are consistently short
            if very_short_ratio > dynamic_threshold or p25_length <= 3:
                objects.mark_dynamic(obj_id)
                num_dynamic += 1
                
                if self.verbose:
                    obj = objects.objects[obj_id]
                    print(f"  Object {obj_id}: {len(stats['track_lengths'])} tracks, "
                          f"p25_length={p25_length:.1f}, short_ratio={very_short_ratio:.2f} -> DYNAMIC")
        
        print(f"Detected {num_dynamic} / {len(objects)} dynamic objects from tracks")
    
    def filter_tracks_by_objects(
        self,
        tracks: Tracks,
        images: Images,
        objects: Objects
    ):
        """Filter tracks based on object consistency and dynamics."""
        
        filter_dynamic_only = self.config.get('filter_dynamic_only', True)
        min_consistency = self.config.get('min_object_consistency', 0.7)
        
        if self.verbose:
            print(f"\nFiltering tracks by object consistency...")
        
        keep_mask = np.ones(len(tracks), dtype=bool)
        num_filtered = 0
        
        dynamic_obj_ids = objects.get_dynamic_objects() if objects and filter_dynamic_only else set()
        
        if self.verbose:
            print(f"Filtering with {len(dynamic_obj_ids)} dynamic objects")
        
        for track_idx in range(len(tracks)):
            track = tracks[track_idx]
            obs = track.observations
            
            if len(obs) == 0:
                continue
            
            # Collect object IDs for all observations
            obj_ids = []
            for img_id, feat_idx in obs:
                if images.semantics[img_id] is None:
                    obj_ids.append(None)
                    continue
                
                feat = images.features[img_id][feat_idx:feat_idx+1]
                label = sample_label_at_points(images.semantics[img_id], feat, mode='center')[0]
                
                if objects:
                    obj_id = objects.get_object_id(img_id, label)
                else:
                    obj_id = (img_id, label) if label > 0 else None
                
                obj_ids.append(obj_id)
            
            # Filter logic
            valid_obj_ids = [oid for oid in obj_ids if oid is not None]
            
            if len(valid_obj_ids) == 0:
                continue
            
            # Find most common object
            unique_ids, counts = np.unique(valid_obj_ids, return_counts=True)
            most_common_obj = unique_ids[np.argmax(counts)]
            consistency = counts[np.argmax(counts)] / len(valid_obj_ids)
            
            # Decide whether to filter
            should_filter = False
            
            if filter_dynamic_only and objects:
                should_filter = most_common_obj in dynamic_obj_ids
            elif not filter_dynamic_only:
                should_filter = consistency < min_consistency
            
            if should_filter:
                keep_mask[track_idx] = False
                num_filtered += 1
        
        # Apply mask to filter tracks
        tracks.filter_by_mask(keep_mask)
        
        if self.verbose:
            print(f"  Kept {np.sum(keep_mask)}/{len(tracks)} tracks")
        else:
            print(f'After object filtering: {len(tracks)} tracks remain ({num_filtered} filtered)')


# Wrapper functions for backward compatibility
def FilterDynamicObjectsPerCamera(view_graph: ViewGraph, images: Images, config: Dict):
    """Detect and filter dynamic objects per-camera (relative pose stage)."""
    semantic_filter = SemanticFilter(config)
    camera_dynamic_labels = semantic_filter.detect_dynamic_per_camera(view_graph, images)
    semantic_filter.filter_matches_by_dynamic_labels(view_graph, images, camera_dynamic_labels)


def AssociateObjectsFromTracks(tracks: Tracks, images: Images, config: Dict) -> Objects:
    """Associate objects across cameras using tracks (track stage)."""
    semantic_filter = SemanticFilter(config)
    objects = semantic_filter.associate_objects_from_tracks(tracks, images)
    
    # Detect which objects are dynamic
    semantic_filter.detect_dynamic_objects_from_tracks(tracks, images, objects)
    
    return objects


def FilterTracksByObjects(tracks: Tracks, images: Images, objects: Objects, config: Dict):
    """Filter tracks based on objects."""
    semantic_filter = SemanticFilter(config)
    semantic_filter.filter_tracks_by_objects(tracks, images, objects)
