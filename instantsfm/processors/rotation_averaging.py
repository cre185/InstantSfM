import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix, csc_matrix, diags
from sksparse.cholmod import cholesky
from queue import Queue
import tqdm

from instantsfm.scene.defs import ViewGraph
from instantsfm.utils.tree import MaximumSpanningTree
from instantsfm.utils.l1_solver import L1Solver
    
class RotationEstimator:
    def __init__(self):
        self.fixed_camera_id = -1

    def InitializeFromMaximumSpanningTree(self, view_graph:ViewGraph, images):
        parents, root = MaximumSpanningTree(view_graph, images)
        children = [[] for _ in range(len(images))]
        for idx, parent in enumerate(parents):
            if idx != root:
                children[parent].append(idx)
        q = Queue()
        q.put(root)

        while not q.empty():
            current = q.get()
            for child in children[current]:
                q.put(child)
            if current == root or parents[current] == -1:
                continue
            # Try both key orders since we don't know which was stored
            parent_id = parents[current]
            pair_key = (current, parent_id) if (current, parent_id) in view_graph.image_pairs else (parent_id, current)
            image_pair = view_graph.image_pairs[pair_key]
            if image_pair.image_id1 == current:
                pair_rotation = R.from_quat(image_pair.rotation).as_matrix()
                rotation = np.linalg.inv(pair_rotation) @ images.world2cams[parents[current], :3, :3]
                images.world2cams[current, :3, :3] = rotation
            else:
                pair_rotation = R.from_quat(image_pair.rotation).as_matrix()
                rotation = pair_rotation @ images.world2cams[parents[current], :3, :3]
                images.world2cams[current, :3, :3] = rotation

    def SetupLinearSystem(self, view_graph:ViewGraph, images):
        registered_mask = images.is_registered
        self.images = {image_id: image for image_id, image in enumerate(images) if registered_mask[image_id]}
        self.image_pairs = {pair_key: pair for pair_key, pair in view_graph.image_pairs.items() if pair.is_valid}
        self.image_id2idx = {}
        self.rotation_estimated = np.zeros(len(images) * 3)
        num_dof = 0
        # Batch compute axis angles for registered images
        Rs = images.world2cams[:, :3, :3]  # (N, 3, 3)
        axis_angles = R.from_matrix(Rs.reshape(-1, 3, 3)).as_rotvec().reshape(len(images), 3)  # (N, 3)
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            self.image_id2idx[image_id] = num_dof
            self.rotation_estimated[num_dof:num_dof+3] = axis_angles[image_id]
            num_dof += 3
            
        if self.fixed_camera_id == -1:
            self.fixed_camera_id = list(self.images.keys())[0]
            self.fixed_camera_rotation = self.images[self.fixed_camera_id].axis_angle()
        
        self.rel_temp_info = {pair_key: {'R_rel': R.from_quat(pair.rotation).as_matrix()} for pair_key, pair in self.image_pairs.items()}
        
        self.sparse_matrix = lil_matrix((len(self.rel_temp_info) * 3 + 3, num_dof))
        curr_pos = 0
        for pair_key, pair in self.image_pairs.items():
            image_id1, image_id2 = pair.image_id1, pair.image_id2
            vector_idx1, vector_idx2 = self.image_id2idx[image_id1], self.image_id2idx[image_id2]
            self.rel_temp_info[pair_key]['index'] = curr_pos
            for i in range(3):
                self.sparse_matrix[curr_pos + i, vector_idx1 + i] = -1
            for i in range(3):
                self.sparse_matrix[curr_pos + i, vector_idx2 + i] = 1
            curr_pos += 3

        for i in range(3):
            self.sparse_matrix[curr_pos + i, self.image_id2idx[self.fixed_camera_id] + i] = 1
        curr_pos += 3

        self.sparse_matrix = self.sparse_matrix.tocsc()

        self.tangent_space_step = np.zeros(num_dof)
        self.tangent_space_residual = np.zeros(curr_pos)

    def ComputeResiduals(self):
        for pair_key, pair_info in self.rel_temp_info.items():
            pair = self.image_pairs[pair_key]
            image_id1, image_id2 = pair.image_id1, pair.image_id2
            idx1, idx2 = self.image_id2idx[image_id1], self.image_id2idx[image_id2]

            R1 = self.rotation_estimated[idx1:idx1+3]
            R1 = R.from_rotvec(R1).as_matrix()
            R2 = self.rotation_estimated[idx2:idx2+3]
            R2 = R.from_rotvec(R2).as_matrix()
            self.tangent_space_residual[pair_info['index']:pair_info['index']+3] = -R.from_matrix(
                R2.T @ pair_info['R_rel'] @ R1).as_rotvec()
        
        self.tangent_space_residual[-3:] = R.from_matrix(R.from_rotvec(self.fixed_camera_rotation).as_matrix().T @ R.from_rotvec(
            self.rotation_estimated[self.image_id2idx[self.fixed_camera_id]:self.image_id2idx[self.fixed_camera_id]+3]).as_matrix()).as_rotvec()

    def SolveL1Regression(self, ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
        self.ComputeResiduals()
        iteration = 0
        curr_norm = 0
        l1_solver = L1Solver(self.sparse_matrix)
        LOCAL_L1_SOLVER_OPTIONS = L1_SOLVER_OPTIONS.copy()
        LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] = 10

        while iteration < ROTATION_ESTIMATOR_OPTIONS['max_num_l1_iterations']:
            last_norm = curr_norm
            self.tangent_space_step = l1_solver.solve(self.tangent_space_residual, LOCAL_L1_SOLVER_OPTIONS)
            if np.any(np.isnan(self.tangent_space_step)):
                print('nan error')
                return False

            curr_norm = np.linalg.norm(self.tangent_space_step)
            self.UpdateGlobalRotations()
            self.ComputeResiduals()

            iteration += 1
            EPS = 1e-6
            if self.ComputeAverageStepSize() < ROTATION_ESTIMATOR_OPTIONS['l1_step_convergence_threshold'] or np.abs(last_norm - curr_norm) < EPS:
                break
            LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] = min(LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] * 2, 100)
        return True

    def SolveIRLS(self, ROTATION_ESTIMATOR_OPTIONS):
        llt = cholesky(csc_matrix(self.sparse_matrix.T @ self.sparse_matrix))

        sigma = np.deg2rad(ROTATION_ESTIMATOR_OPTIONS['irls_loss_parameter_sigma'])

        weights_irls = np.ones(self.sparse_matrix.shape[0])

        self.ComputeResiduals()
        for _ in range(ROTATION_ESTIMATOR_OPTIONS['max_num_irls_iterations']):
            for pair_info in self.rel_temp_info.values():
                image_pair_pos = pair_info['index']
                err_squared = 0
                err_squared = np.sum(self.tangent_space_residual[image_pair_pos:image_pair_pos+3]**2)

                # use GEMAN_MCCLURE loss function
                tmp = err_squared + sigma**2
                w = sigma**2 / (tmp**2)

                if np.isnan(w):
                    print("nan weight!")
                    return False

                weights_irls[image_pair_pos:image_pair_pos+3] = w

            at_weight = self.sparse_matrix.T @ diags(weights_irls)

            llt = cholesky(csc_matrix(at_weight @ self.sparse_matrix))

            self.tangent_space_step.fill(0)
            self.tangent_space_step = llt(at_weight @ self.tangent_space_residual)
            self.UpdateGlobalRotations()
            self.ComputeResiduals()

            if self.ComputeAverageStepSize() < ROTATION_ESTIMATOR_OPTIONS['irls_step_convergence_threshold']:
                break
        return True

    def UpdateGlobalRotations(self):
        for image_id in self.images.keys():
            vector_idx = self.image_id2idx[image_id]
            R_ori = R.from_rotvec(self.rotation_estimated[vector_idx:vector_idx+3]).as_matrix()
            self.rotation_estimated[vector_idx:vector_idx+3] = R.from_matrix(
                R_ori @ R.from_rotvec(-self.tangent_space_step[vector_idx:vector_idx+3]).as_matrix()).as_rotvec()

    def ComputeAverageStepSize(self):
        total_update = 0.
        for image_id in self.images.keys():
            vector_idx = self.image_id2idx[image_id]
            total_update += np.linalg.norm(self.tangent_space_step[vector_idx:vector_idx+3])
        return total_update / len(self.image_id2idx)

    def EstimateRotations(self, view_graph:ViewGraph, images, ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
        self.InitializeFromMaximumSpanningTree(view_graph, images)
        self.SetupLinearSystem(view_graph, images)

        if ROTATION_ESTIMATOR_OPTIONS['max_num_l1_iterations'] > 0:
            if not self.SolveL1Regression(ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
                return False
            
        if ROTATION_ESTIMATOR_OPTIONS['max_num_irls_iterations'] > 0:
            if not self.SolveIRLS(ROTATION_ESTIMATOR_OPTIONS):
                return False
            
        # Batch write rotation results
        for image_id in range(len(images)):
            if not images.is_registered[image_id]:
                continue
            idx = self.image_id2idx[image_id]
            images.world2cams[image_id, :3, :3] = R.from_rotvec(self.rotation_estimated[idx:idx+3]).as_matrix()
        view_graph.image_pairs.update(self.image_pairs)
        return True