import time
import numpy as np
import random

from instantsfm.scene.defs import ViewGraph
from instantsfm.controllers.config import Config
from instantsfm.processors.view_graph_manipulation import UpdateImagePairsConfig, DecomposeRelPose
from instantsfm.processors.view_graph_calibration import SolveViewGraphCalibration, TorchVGC
from instantsfm.processors.image_undistortion import UndistortImages
from instantsfm.processors.relpose_estimation import EstimateRelativePose
from instantsfm.processors.image_pair_inliers import ImagePairInliersCount
from instantsfm.processors.relpose_filter import FilterInlierNum, FilterInlierRatio, FilterRotations
from instantsfm.processors.rotation_averaging import RotationEstimator
from instantsfm.processors.track_establishment import TrackEngine
from instantsfm.processors.reconstruction_normalizer import NormalizeReconstruction
from instantsfm.processors.global_positioning import TorchGP
from instantsfm.processors.track_filter import FilterTracksByAngle, FilterTracksByReprojectionNormalized, FilterTracksTriangulationAngle
from instantsfm.processors.bundle_adjustment import TorchBA
from instantsfm.processors.track_retriangulation import RetriangulateTracks
from instantsfm.processors.reconstruction_pruning import PruneWeaklyConnectedImages
from instantsfm.processors.semantic_filtering import FilterDynamicObjectsPerCamera, AssociateObjectsFromTracks, FilterTracksByObjects
from instantsfm.scene.rig_utils import Single2Rig, Rig2Single


def SolveGlobalMapper(view_graph:ViewGraph, cameras, images, config:Config, visualizer=None):
    seed = config.RUNTIME_OPTIONS.get('random_seed', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # Global objects for cross-camera analysis (initialized after tracks)
    global_objects = None
    
    if not config.OPTIONS['skip_preprocessing']:
        print('-------------------------------------')
        print('Running preprocessing ...')
        print('-------------------------------------')
        start_time = time.time()
        UpdateImagePairsConfig(view_graph, cameras, images)
        DecomposeRelPose(view_graph, cameras, images)
        print('Preprocessing took: ', time.time() - start_time)

    # disable if in multi-camera rig mode: most likely the intrinsics are already well calibrated
    if not config.OPTIONS['skip_view_graph_calibration'] and not config.RUNTIME_OPTIONS['multi_camera_rig']:
        print('-------------------------------------')
        print('Running view graph calibration ...')
        print('-------------------------------------')
        start_time = time.time()
        # vgc_engine = TorchVGC()
        # vgc_engine.Optimize(view_graph, cameras, images, config.VIEW_GRAPH_CALIBRATOR_OPTIONS)
        SolveViewGraphCalibration(view_graph, cameras, images, config.VIEW_GRAPH_CALIBRATOR_OPTIONS)
        print('View graph calibration took: ', time.time() - start_time)
        
    print('-------------------------------------')
    print('Running relative pose estimation ...')
    print('-------------------------------------')
    start_time = time.time()
    UndistortImages(cameras, images)
    use_poselib = False
    if use_poselib:
        EstimateRelativePose(view_graph, cameras, images, use_poselib)
        ImagePairInliersCount(view_graph, cameras, images, config.INLIER_THRESHOLD_OPTIONS)
    else:
        EstimateRelativePose(view_graph, cameras, images)
    
    FilterInlierNum(view_graph, config.INLIER_THRESHOLD_OPTIONS['min_inlier_num'])
    FilterInlierRatio(view_graph, config.INLIER_THRESHOLD_OPTIONS['min_inlier_ratio'])
    
    if config.RUNTIME_OPTIONS['use_semantic_filtering']:
        FilterDynamicObjectsPerCamera(view_graph, images, config.SEMANTIC_OPTIONS)
    
    view_graph.keep_largest_connected_component(images)
    # check how many images are left
    print('Relative pose estimation took: ', time.time() - start_time)

    print('-------------------------------------')
    print('Running rotation averaging ...')
    print('-------------------------------------')
    start_time = time.time()
    ra_engine = RotationEstimator()
    ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS)
    FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
    if not view_graph.keep_largest_connected_component(images):
        print('Failed to keep the largest connected component.')
        exit()

    ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS)
    FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
    if not view_graph.keep_largest_connected_component(images):
        print('Failed to keep the largest connected component.')
        exit()
    num_img = np.sum(images.is_registered)
    print(num_img, '/', len(images), 'images are within the connected component.')
    print('Rotation estimation took: ', time.time() - start_time)

    print('-------------------------------------')
    print('Running track establishment ...')
    print('-------------------------------------')
    start_time = time.time()
    track_engine = TrackEngine(view_graph, images)
    tracks_orig = track_engine.EstablishFullTracks(config.TRACK_ESTABLISHMENT_OPTIONS)
    print('Initialized', len(tracks_orig), 'tracks')

    tracks = track_engine.FindTracksForProblem(tracks_orig, config.TRACK_ESTABLISHMENT_OPTIONS)
    print('Before filtering:', len(tracks_orig), ', after filtering:', len(tracks))
    
    if config.RUNTIME_OPTIONS['use_semantic_filtering']:
        global_objects = AssociateObjectsFromTracks(tracks, images, config.SEMANTIC_OPTIONS)
        FilterTracksByObjects(tracks, images, global_objects, config.SEMANTIC_OPTIONS)
    
    print('Track establishment took: ', time.time() - start_time)

    print('-------------------------------------')
    print('Running global positioning ...')
    print('-------------------------------------')
    start_time = time.time()
    UndistortImages(cameras, images)
    gp_engine = TorchGP(visualizer=visualizer)
    gp_use_rig = config.RUNTIME_OPTIONS['multi_camera_rig']
    if gp_use_rig:
        Single2Rig(images, cameras, tracks, config.VOTING_OPTIONS, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
    gp_engine.InitializeRandomPositions(cameras, images, tracks, 
                                        is_multi=gp_use_rig, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
    gp_engine.Optimize(cameras, images, tracks, config.GLOBAL_POSITIONER_OPTIONS, use_depths=config.RUNTIME_OPTIONS['use_depths'], 
                       is_multi=gp_use_rig, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
    tracks = FilterTracksByAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_angle_error'])
    NormalizeReconstruction(images, tracks, use_depths=config.RUNTIME_OPTIONS['use_depths'])
    print('Global positioning took: ', time.time() - start_time)

    print('-------------------------------------')
    print('Running bundle adjustment ...')
    print('-------------------------------------')
    start_time = time.time()
    ba_engine = TorchBA(visualizer=visualizer)
    ba_use_rig = config.RUNTIME_OPTIONS['multi_camera_rig'] # and config.RUNTIME_OPTIONS['use_fixed_rel_poses']
    if ba_use_rig:
        Single2Rig(images, cameras, tracks, config.VOTING_OPTIONS, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
        
    for iter in range(3):
        ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS, use_depths=config.RUNTIME_OPTIONS['use_depths'], 
                        is_multi=ba_use_rig, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
        UndistortImages(cameras, images)
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter))
    
    print(f'{np.sum(images.is_registered)} images are registered after BA.')
        
    print('Filtering tracks')
    UndistortImages(cameras, images)
    FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
    FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
    NormalizeReconstruction(images, tracks, use_depths=config.RUNTIME_OPTIONS['use_depths'])
    
    if config.RUNTIME_OPTIONS['multi_camera_rig']:
        print('Rebuilding rig structure after BA...')
        Single2Rig(images, cameras, tracks, config.VOTING_OPTIONS, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
    
    print('Bundle adjustment took: ', time.time() - start_time)

    if not config.OPTIONS['skip_retriangulation']:
        print('-------------------------------------')
        print('Running retriangulation ...')
        print('-------------------------------------')
        start_time = time.time()
        RetriangulateTracks(cameras, images, tracks, tracks_orig, config.TRIANGULATOR_OPTIONS, config.BUNDLE_ADJUSTER_OPTIONS)

        print('-------------------------------------')
        print('Running bundle adjustment ...')
        print('-------------------------------------')
        ba_engine = TorchBA()
        ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS, use_depths=config.RUNTIME_OPTIONS['use_depths'])

        # NormalizeReconstruction(images, tracks)
        UndistortImages(cameras, images)
        print('Filtering tracks')
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        print('Retriangulation took: ', time.time() - start_time)

    if not config.OPTIONS['skip_pruning']:
        print('-------------------------------------')
        print('Running postprocessing ...')
        print('-------------------------------------')
        start_time = time.time()
        PruneWeaklyConnectedImages(images, tracks)
        print('Postprocessing took: ', time.time() - start_time)

    return cameras, images, tracks
