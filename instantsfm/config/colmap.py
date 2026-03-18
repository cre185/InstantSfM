CONFIG = {
    # SfM pipeline configs
    'VIEW_GRAPH_CALIBRATOR_OPTIONS': {
        'thres_lower_ratio': 0.1,
        'thres_higher_ratio': 10,
        'thres_two_view_error': 2.,
        'thres_loss_function': 1e-2,
        'max_num_iterations': 100,
        'function_tolerance': 5e-4,
    },
    'INLIER_THRESHOLD_OPTIONS': {
        'max_angle_error': 1.,
        'max_reprojection_error': 1e-2,
        'min_triangulation_angle': 1.,
        'max_epipolar_error_E': 1.,
        'max_epipolar_error_F': 4.,
        'max_epipolar_error_H': 4.,
        'min_inlier_num': 30,
        'min_inlier_ratio': 0.25,
        'max_rotation_error': 10.,
    },
    'SEMANTIC_OPTIONS': {
        'min_observations_for_dynamic': 3,  # Minimum pairs to judge dynamic (per-camera)
        'dynamic_inlier_threshold': 0.3,  # p25 inlier ratio threshold
        'use_percentile': 25,  # Use p25 for dynamic detection
        'min_observations_for_object': 3,  # Minimum track observations for object
        'min_object_consistency': 0.7,  # Minimum label consistency in track
        'track_length_ratio_threshold': 0.5,  # Ratio of short tracks indicating dynamic
        'filter_dynamic_only': True,  # Only filter dynamic objects
        'verbose_semantics': False,  # Print detailed statistics
    },
    'ROTATION_ESTIMATOR_OPTIONS': {
        'max_num_l1_iterations': 10,
        'l1_step_convergence_threshold': 0.001,
        'max_num_irls_iterations': 100,
        'irls_step_convergence_threshold': 0.001,
        'irls_loss_parameter_sigma': 5.0,
    },
    'L1_SOLVER_OPTIONS': {
        'max_num_iterations': 1000,
        'rho': 1.0,
        'alpha': 1.0,
        'absolute_tolerance': 1e-4,
        'relative_tolerance': 1e-2,
    },
    'TRACK_ESTABLISHMENT_OPTIONS': {
        'thres_inconsistency': 10.,
        'min_num_view_per_track': 3,
        'max_num_view_per_track': 200,
    },
    'GLOBAL_POSITIONER_OPTIONS': {
        'min_num_view_per_track': 3,
        'thres_loss_function': 1e-1,
        'max_num_iterations': 100,
        'function_tolerance': 5e-4,
    },
    'BUNDLE_ADJUSTER_OPTIONS': {
        'optimize_poses': True,
        'optimize_points': True,
        'optimize_intrinsics': True,
        'min_num_view_per_track': 2,
        'thres_loss_function': 1.,
        'max_num_iterations': 100,
        'function_tolerance': 5e-4,
        'depth_weight': 0.1,
    },
    'TRIANGULATOR_OPTIONS': {
        'min_num_view_per_track': 2,
        'complete_max_reproj_error': 3.0,
        'merge_max_reproj_error': 3.0,
        'filter_max_reproj_error': 3.0,
        'filter_min_tri_angle': 1.5,
        'ba_global_max_refinements': 5,
        'ba_global_max_refinement_change': 0.0005,
    },
    'VOTING_OPTIONS': {
        'initial_angle_threshold': 5.0,
        'target_party_ratio': 0.25,
        'angle_threshold_step': 0.5,
        'min_angle_threshold': 2.0,
        'max_angle_threshold': 10.0,
        'initial_distance_threshold': 0.5,
        'distance_threshold_step': 0.1,
        'min_distance_threshold': 0.1,
        'max_distance_threshold': 1.0,
    },
    
    # feature handler configs
    'FEATURE_HANDLER_OPTIONS': {
        'min_num_matches': 30,
        'uniform_camera': True,
    },
}
