import numpy as np
import os

from instantsfm.scene.reconstruction import Reconstruction

def ExportReconstruction(output_path, cameras, images, tracks, image_path, cluster_id=-1, include_image_points=False, export_txt=False):
    """Export reconstruction using Images/Tracks containers directly.
    
    Args:
        output_path: Output directory path
        cameras: List of camera models
        images: Images container
        tracks: Tracks container
        image_path: Path to image directory for color extraction
        cluster_id: Cluster ID to filter (-1 for all)
        include_image_points: Whether to include image points
        export_txt: Export as text format instead of binary
    """
    # Create reconstruction with containers
    reconstruction = Reconstruction(cameras=cameras, images=images, tracks=tracks)
    
    # Filter by cluster if needed
    if cluster_id != -1:
        reconstruction.filter_by_cluster(cluster_id)
    
    # Filter by registration status
    reconstruction.filter_registered_only()
    
    # Build 2D-3D correspondences
    reconstruction.build_correspondences(min_track_length=3 if not include_image_points else 0)
    
    # Extract colors from images if path provided
    if image_path != "":
        reconstruction.extract_colors_batch(image_path)
    
    # Create output directory
    cluster_path = os.path.join(output_path, '0' if cluster_id == -1 else str(cluster_id))
    os.makedirs(cluster_path, exist_ok=True)

    # Write reconstruction
    if export_txt:
        reconstruction.write_text(cluster_path)
    else:
        reconstruction.write_binary(cluster_path)
    
    print(f'Exported {reconstruction.num_points} points and {reconstruction.num_images} images')

def FilterRigCompleteness(images):
    """Filter images by rig validity for multi-camera rigs.
    
    For multi-camera rigs, a rig group is considered valid if at least one camera
    in the group is registered. This function keeps all registered images from
    valid rigs (or standalone images).
    
    Args:
        images: Images container with rig information
    
    Returns:
        Boolean mask indicating which images to keep
    """
    if images.rig_groups.size == 0:
        # No rig structure, keep all registered images
        return images.is_registered.copy()
    
    num_groups = images.rig_groups.shape[0]
    
    # Check each rig group for validity - at least one registered camera
    rig_has_registered = np.zeros(num_groups, dtype=bool)
    for group_idx in range(num_groups):
        rig_image_indices = images.rig_groups[group_idx]
        valid_cameras = rig_image_indices[rig_image_indices != -1]
        
        # A rig is valid if at least one camera is registered
        for img_idx in valid_cameras:
            if images.is_registered[img_idx]:
                rig_has_registered[group_idx] = True
                break
    
    # Keep registered images from valid rigs
    keep_mask = images.is_registered.copy()
    for i in range(len(images)):
        group_idx = images.image_to_rig[i, 0]
        if group_idx != -1:
            # Part of a rig - keep only if individually registered AND rig has at least one registered
            keep_mask[i] = images.is_registered[i] and rig_has_registered[group_idx]
        # Standalone images keep their individual registered status
    
    return keep_mask


def WriteGlomapReconstruction(output_path, cameras, images, tracks, image_path, export_txt=False):
    """Write reconstruction, handling multiple clusters if present.
    
    Args:
        output_path: Output directory path
        cameras: List of camera models
        images: Images container
        tracks: Tracks container
        image_path: Path to image directory
        export_txt: Export as text format instead of binary
    """
    # Apply rig completeness filtering
    original_registered = images.is_registered.copy()
    rig_filtered = FilterRigCompleteness(images)
    
    # Temporarily update registration status for export
    filtered_count = np.sum(rig_filtered)
    removed_count = np.sum(original_registered) - filtered_count
    if removed_count > 0:
        print(f"Rig completeness filtering: removed {removed_count} images from incomplete rigs")
    
    images.is_registered = rig_filtered
    
    try:
        # Check for clusters
        if not hasattr(images, 'cluster_ids'):
            # No clustering, export all
            ExportReconstruction(output_path, cameras, images, tracks, image_path, export_txt=export_txt)
            return
        
        # Find max cluster ID
        registered_mask = images.is_registered
        if not np.any(registered_mask):
            print("No registered images to export")
            return
        
        cluster_ids = images.cluster_ids[registered_mask]
        unique_clusters = np.unique(cluster_ids)
        
        if len(unique_clusters) == 1:
            # Single cluster, export all
            ExportReconstruction(output_path, cameras, images, tracks, image_path, export_txt=export_txt)
        else:
            # Multiple clusters, export separately
            for cluster_id in unique_clusters:
                print(f'Exporting reconstruction for cluster {cluster_id} ({np.sum(cluster_ids == cluster_id)} images)')
                ExportReconstruction(output_path, cameras, images, tracks, 
                                     image_path, cluster_id, export_txt=export_txt)
    finally:
        # Restore original registration status
        images.is_registered = original_registered