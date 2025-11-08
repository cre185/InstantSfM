from collections import defaultdict
import numpy as np

from instantsfm.utils.union_find import UnionFind
from instantsfm.scene.defs import Track, ViewGraph

class TrackEngine:

    def __init__(self, view_graph:ViewGraph, images):
        self.view_graph = view_graph
        self.images = images
        self.uf = UnionFind()

    def EstablishFullTracks(self, TRACK_ESTABLISHMENT_OPTIONS):
        import time
        start_time = time.time()
        self.BlindConcatenation()
        print(f"Blind concatenation took {time.time() - start_time} seconds")
        tracks = self.TrackCollection(TRACK_ESTABLISHMENT_OPTIONS)
        print(f"Track collection took {time.time() - start_time} seconds")
        return tracks

    def BlindConcatenation(self):
        for pair in self.view_graph.image_pairs.values():
            if not pair.is_valid:
                continue
            matches = pair.matches
            for idx in pair.inliers:
                point1, point2 = matches[idx]

                point_global_id1 = (int(pair.image_id1) << 32) | point1
                point_global_id2 = (int(pair.image_id2) << 32) | point2
                
                if point_global_id2 < point_global_id1:
                    self.uf.Union(point_global_id1, point_global_id2)
                else:
                    self.uf.Union(point_global_id2, point_global_id1)
    
    def TrackCollection(self, TRACK_ESTABLISHMENT_OPTIONS):
        track_map = {}
        for pair in self.view_graph.image_pairs.values():
            if not pair.is_valid:
                continue
            for idx in pair.inliers:
                point1, point2 = pair.matches[idx]

                point_global_id1 = (pair.image_id1 << 32) | point1
                
                track_id = self.uf.Find(point_global_id1)

                if track_id not in track_map:
                    track_map[track_id] = defaultdict(int)  # this is the reference counter
                track_map[track_id][(pair.image_id1, point1)] += 1
                track_map[track_id][(pair.image_id2, point2)] += 1

        tracks = {track_id: np.concatenate([np.array(list(correspondences.keys())), 
                                            -np.array(list(correspondences.values()))[:, None]], axis=-1) 
                                            for track_id, correspondences in track_map.items()}
        discarded_counter = 0
        for track_id in list(tracks.keys()):
            # verify consistency of observations
            image_id_set = {}
            for image_id, feature_id, _ in tracks[track_id]:
                image_feature = self.images[image_id].features[feature_id]
                if image_id not in image_id_set:
                    image_id_set[image_id] = image_feature.reshape(1, 2)
                else:
                    features_array = image_id_set[image_id]
                    distances = np.linalg.norm(features_array - image_feature, axis=1)
                    if np.any(distances > TRACK_ESTABLISHMENT_OPTIONS['thres_inconsistency']):
                        del tracks[track_id]
                        discarded_counter += 1
                        break
                    image_id_set[image_id] = np.vstack([features_array, image_feature.reshape(1, 2)])
            if track_id not in tracks:
                continue
            
            # filter out multiple observations in the same image
            correspondences = tracks[track_id]
            sort_by_prio, unique_indices = np.unique(correspondences[:, [0, 2]], axis=0, return_index=True)
            unique_image_ids, unique_indices_ = np.unique(sort_by_prio[:, 0], return_index=True)
            discarded_counter += len(correspondences) - len(unique_indices_)
            tracks[track_id] = correspondences[unique_indices[unique_indices_], :2]

        print(f"Discarded {discarded_counter} features due to deduplication")
        return tracks
    
    def FindTracksForProblem(self, tracks_full, TRACK_ESTABLISHMENT_OPTIONS):
        tracks_per_camera = {}
        tracks = {}
        for image_id, image in enumerate(self.images):
            if not image.is_registered:
                continue
            tracks_per_camera[image_id] = 0
        # if input image resolution is too low, TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track'] is suggested to be small, e.g. 1 or 2.... to make sure all image indices are included
        # TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track'] = 2
        for track_id, track_obs in tracks_full.items():
            if track_obs.shape[0] < TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track']:
                continue
            if track_obs.shape[0] > TRACK_ESTABLISHMENT_OPTIONS['max_num_view_per_track']:
                continue
            track_obs = tracks_full[track_id]
            track_temp = Track(id=track_id)
            track_temp.observations = track_obs[np.isin(track_obs[:, 0], list(tracks_per_camera.keys()))]
            tracks[track_id] = track_temp
        
        return tracks