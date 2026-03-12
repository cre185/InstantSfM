import time
import numpy as np

from instantsfm.scene.defs import ViewGraph, Tracks
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


def _find_valid_components(view_graph: ViewGraph, num_images: int, active_mask=None):
    adjacency = {}
    for pair in view_graph.image_pairs.values():
        if not pair.is_valid:
            continue
        if active_mask is not None and (not active_mask[pair.image_id1] or not active_mask[pair.image_id2]):
            continue
        adjacency.setdefault(pair.image_id1, set()).add(pair.image_id2)
        adjacency.setdefault(pair.image_id2, set()).add(pair.image_id1)

    visited = np.zeros(num_images, dtype=bool)
    components = []
    for image_id in adjacency.keys():
        if visited[image_id]:
            continue
        queue = [image_id]
        visited[image_id] = True
        component = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if component:
            component.sort()
            components.append(component)

    components.sort(key=len, reverse=True)
    return components


def _component_mask(num_images: int, component):
    mask = np.zeros(num_images, dtype=bool)
    mask[np.asarray(component, dtype=np.int32)] = True
    return mask


def _invalidate_component_pairs(view_graph: ViewGraph, component_mask):
    for pair in view_graph.image_pairs.values():
        if component_mask[pair.image_id1] and component_mask[pair.image_id2]:
            pair.is_valid = False


def _append_tracks(dst_tracks: Tracks, src_tracks: Tracks):
    for track_id in range(len(src_tracks)):
        dst_tracks.append(
            id=int(src_tracks.ids[track_id]),
            xyz=src_tracks.xyzs[track_id].copy(),
            color=src_tracks.colors[track_id].copy(),
            is_initialized=bool(src_tracks.is_initialized[track_id]),
            observations=src_tracks.observations[track_id].copy(),
            semantic_label=int(src_tracks.semantic_labels[track_id]),
            semantic_confidence=float(src_tracks.semantic_confidences[track_id]),
        )


def _optimize_registered_component(view_graph, cameras, images, tracks_orig, track_engine, config: Config, visualizer=None, component_id=None):
    component_label = f'component {component_id}' if component_id is not None else 'component'
    tracks = track_engine.FindTracksForProblem(tracks_orig, config.TRACK_ESTABLISHMENT_OPTIONS)
    print(f'Initialized {len(tracks)} tracks for {component_label}')

    if len(tracks) == 0:
        return tracks

    if config.RUNTIME_OPTIONS['use_semantic_filtering']:
        global_objects = AssociateObjectsFromTracks(tracks, images, config.SEMANTIC_OPTIONS)
        FilterTracksByObjects(tracks, images, global_objects, config.SEMANTIC_OPTIONS)

    print('-------------------------------------')
    print(f'Running global positioning for {component_label} ...')
    print('-------------------------------------')
    start_time = time.time()
    UndistortImages(cameras, images)
    gp_engine = TorchGP(visualizer=visualizer)
    gp_use_rig = config.RUNTIME_OPTIONS['multi_camera_rig']
    random_seed = config.RUNTIME_OPTIONS.get('random_seed')
    if random_seed is not None:
        np.random.seed(random_seed)
    if gp_use_rig:
        Single2Rig(images, cameras, tracks, config.VOTING_OPTIONS, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])
    gp_engine.InitializeRandomPositions(
        cameras, images, tracks,
        is_multi=gp_use_rig, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses']
    )
    gp_engine.Optimize(
        cameras, images, tracks, config.GLOBAL_POSITIONER_OPTIONS,
        use_depths=config.RUNTIME_OPTIONS['use_depths'],
        is_multi=gp_use_rig, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses']
    )
    tracks = FilterTracksByAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_angle_error'])
    NormalizeReconstruction(images, tracks, use_depths=config.RUNTIME_OPTIONS['use_depths'])
    print('Global positioning took: ', time.time() - start_time)

    print('-------------------------------------')
    print(f'Running bundle adjustment for {component_label} ...')
    print('-------------------------------------')
    start_time = time.time()
    ba_engine = TorchBA(visualizer=visualizer)
    ba_use_rig = config.RUNTIME_OPTIONS['multi_camera_rig']
    if ba_use_rig:
        Single2Rig(images, cameras, tracks, config.VOTING_OPTIONS, use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'])

    for iter in range(3):
        if not ba_use_rig:
            ba_engine.Solve(
                cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS,
                use_depths=config.RUNTIME_OPTIONS['use_depths'],
                is_multi=False,
                fix_rotation=True,
            )
        ba_engine.Solve(
            cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS,
            use_depths=config.RUNTIME_OPTIONS['use_depths'],
            is_multi=ba_use_rig,
            use_fixed_rel_poses=config.RUNTIME_OPTIONS['use_fixed_rel_poses'],
        )
        UndistortImages(cameras, images)
        FilterTracksByReprojectionNormalized(
            cameras, images, tracks,
            config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter)
        )

    print(f'{np.sum(images.is_registered)} images are registered after BA in {component_label}.')

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
        print(f'Running retriangulation for {component_label} ...')
        print('-------------------------------------')
        start_time = time.time()
        tracks = RetriangulateTracks(
            view_graph,
            cameras,
            images,
            tracks,
            tracks_orig,
            config.TRIANGULATOR_OPTIONS,
            config.BUNDLE_ADJUSTER_OPTIONS,
        )

        print('-------------------------------------')
        print(f'Running bundle adjustment for {component_label} ...')
        print('-------------------------------------')
        ba_engine = TorchBA()
        if not config.RUNTIME_OPTIONS['multi_camera_rig']:
            ba_engine.Solve(
                cameras,
                images,
                tracks,
                config.BUNDLE_ADJUSTER_OPTIONS,
                use_depths=config.RUNTIME_OPTIONS['use_depths'],
                fix_rotation=True,
            )
        ba_engine.Solve(
            cameras,
            images,
            tracks,
            config.BUNDLE_ADJUSTER_OPTIONS,
            use_depths=config.RUNTIME_OPTIONS['use_depths'],
        )

        UndistortImages(cameras, images)
        print('Filtering tracks')
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        print('Retriangulation took: ', time.time() - start_time)

    return tracks


def SolveGlobalMapper(view_graph:ViewGraph, cameras, images, config:Config, visualizer=None):
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

    print('Relative pose estimation took: ', time.time() - start_time)

    print('-------------------------------------')
    print('Running rotation averaging ...')
    print('-------------------------------------')
    start_time = time.time()
    preserve_all_connected_components = (
        config.OPTIONS.get('preserve_all_connected_components', False)
        and not config.RUNTIME_OPTIONS['multi_camera_rig']
    )
    if not preserve_all_connected_components:
        if not view_graph.keep_largest_connected_component(images):
            print('Failed to keep the largest connected component.')
            return cameras, images, Tracks()

        ra_engine = RotationEstimator()
        if not ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS):
            return cameras, images, Tracks()
        FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
        if not view_graph.keep_largest_connected_component(images):
            print('Failed to keep the largest connected component.')
            return cameras, images, Tracks()

        ra_engine = RotationEstimator()
        if not ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS):
            return cameras, images, Tracks()
        FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
        if not view_graph.keep_largest_connected_component(images):
            print('Failed to keep the largest connected component.')
            return cameras, images, Tracks()
        final_components = [images.get_registered_indices().tolist()]
    else:
        first_pass_components = []
        initial_components = _find_valid_components(view_graph, len(images))
        for component_id, component in enumerate(initial_components):
            component_mask = _component_mask(len(images), component)
            images.is_registered[:] = component_mask
            print(f'Rotation averaging pass 1 on component {component_id} ({len(component)} images)')
            ra_engine = RotationEstimator()
            if not ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS):
                _invalidate_component_pairs(view_graph, component_mask)
                continue
            FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
            first_pass_components.extend(_find_valid_components(view_graph, len(images), component_mask))

        final_components = []
        for component_id, component in enumerate(first_pass_components):
            component_mask = _component_mask(len(images), component)
            images.is_registered[:] = component_mask
            print(f'Rotation averaging pass 2 on component {component_id} ({len(component)} images)')
            ra_engine = RotationEstimator()
            if not ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS):
                _invalidate_component_pairs(view_graph, component_mask)
                continue
            FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
            final_components.extend(_find_valid_components(view_graph, len(images), component_mask))

        final_components = [component for component in final_components if len(component) >= 2]
        final_components.sort(key=len, reverse=True)
        images.is_registered[:] = False
        images.cluster_ids[:] = -1
        for cluster_id, component in enumerate(final_components):
            images.is_registered[component] = True
            images.cluster_ids[component] = cluster_id

    num_img = np.sum(images.is_registered)
    print(num_img, '/', len(images), 'images are within the solved components.')
    print('Rotation estimation took: ', time.time() - start_time)

    if len(final_components) == 0:
        return cameras, images, Tracks()

    print('-------------------------------------')
    print('Running track establishment ...')
    print('-------------------------------------')
    start_time = time.time()
    track_engine = TrackEngine(view_graph, images)
    tracks_orig = track_engine.EstablishFullTracks(config.TRACK_ESTABLISHMENT_OPTIONS)
    print('Initialized', len(tracks_orig), 'tracks')
    print('Track establishment took: ', time.time() - start_time)

    solved_tracks = Tracks()
    solved_mask = np.zeros(len(images), dtype=bool)
    for component_id, component in enumerate(final_components):
        print('-------------------------------------')
        print(f'Solving component {component_id} ({len(component)} images)')
        print('-------------------------------------')
        component_mask = _component_mask(len(images), component)
        images.is_registered[:] = component_mask
        tracks = _optimize_registered_component(
            view_graph, cameras, images, tracks_orig, track_engine, config, visualizer=visualizer, component_id=component_id
        )
        if len(tracks) == 0:
            continue
        solved_mask |= images.is_registered
        _append_tracks(solved_tracks, tracks)

    images.is_registered[:] = solved_mask

    if not config.OPTIONS['skip_pruning']:
        if len(final_components) == 1:
            print('-------------------------------------')
            print('Running postprocessing ...')
            print('-------------------------------------')
            start_time = time.time()
            PruneWeaklyConnectedImages(images, solved_tracks)
            print('Postprocessing took: ', time.time() - start_time)
        else:
            print('Skipping postprocessing because multi-component pruning still collapses to a single component.')

    return cameras, images, solved_tracks
