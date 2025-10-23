import numpy as np
import torch
import tqdm
import pyceres

from instantsfm.scene.defs import ConfigurationType, ViewGraph
from instantsfm.utils.cost_function import FetzerFocalLengthCostFunction, FetzerFocalLengthSameCameraCostFunction, fetzer_ds, fetzer_cost

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Cauchy
from bae.utils.pysolvers import PCG
from bae.optim import LM
from bae.autograd.function import TrackingTensor

def SolveViewGraphCalibration(view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
    valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                         if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
    focals = np.array([np.mean(cam.focal_length) for cam in cameras])

    problem = pyceres.Problem()
    options = pyceres.SolverOptions()
    loss_function = pyceres.CauchyLoss(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
    if len(cameras) < 50:
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_NORMAL_CHOLESKY
    else:
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    
    for image_pair in valid_image_pairs.values():
        image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
        cam1, cam2 = cameras[image1.cam_id], cameras[image2.cam_id]
        if cam1 == cam2:
            cost_function = FetzerFocalLengthSameCameraCostFunction(image_pair.F, cam1.principal_point)
            idx = image1.cam_id
            problem.add_residual_block(cost_function, loss_function, [focals[idx:idx+1]])
        else:
            cost_function = FetzerFocalLengthCostFunction(image_pair.F, cam1.principal_point, cam2.principal_point)
            idx1, idx2 = image1.cam_id, image2.cam_id
            problem.add_residual_block(cost_function, loss_function, [focals[idx1:idx1+1], focals[idx2:idx2+1]])
    problem.set_parameter_lower_bound(focals, 0, 1e-3)

    options.max_num_iterations = VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations']
    options.function_tolerance = VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']
    # options.minimizer_progress_to_stdout = True

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())

    # copy back results
    counter = 0
    for idx, cam in enumerate(cameras):
        focal = focals[idx]
        if (focal / np.mean(cam.focal_length) < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio'] or 
            focal / np.mean(cam.focal_length) > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']):
            counter += 1
            continue
        cam.has_refined_focal_length = True
        cam.focal_length = np.array([focal, focal])
    
    print(f'{counter} cameras are rejected in view graph calibration')

    # Filter Image Pairs
    eval_options = pyceres.EvaluateOptions()
    eval_options.apply_loss_function = False
    residuals = problem.evaluate_residuals(eval_options)
    invalid_counter = 0
    thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

    # manually calculate the residuals
    for idx, (pair_id, image_pair) in enumerate(valid_image_pairs.items()):
        cost_function.Evaluate([cam1.focal_length, cam2.focal_length], residuals, None)
        residual = residuals[2 * idx:2 * idx + 2]
        if residual[0]**2 + residual[1]**2 > thres_two_view_error_sq:
            invalid_counter += 1
            image_pair.is_valid = False
            view_graph.image_pairs[pair_id].is_valid = False
    print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')

class TorchVGC():
    def __init__(self, device='cuda:0'):
        self.device = device

    def Optimize(self, view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
        cost_fn = fetzer_cost
        class FetzerNonBatched(nn.Module):
            def __init__(self, focals):
                super().__init__()
                self.focals = nn.Parameter(TrackingTensor(focals)) # (num_cameras, 1)
            def forward(self, ds, camera_indices1, camera_indices2):
                # ds: (num_pairs, 1, 3, 4)
                loss = cost_fn(self.focals[camera_indices1], self.focals[camera_indices2], ds)
                return loss

        valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                             if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
        focals = torch.tensor(np.array([np.mean(cam.focal_length) for cam in cameras]), dtype=torch.float64).to(self.device).unsqueeze(-1)
        self.camera_has_prior = torch.tensor([cam.has_prior_focal_length for cam in cameras], dtype=torch.bool).to(self.device)
        # TODO: Only support all cameras have prior focal length. If some cameras have prior focal length while others do not, 
        # they will be optimized together, which is not a good idea.
        if torch.all(self.camera_has_prior):
            print('All cameras have prior focal length, skipping view graph calibration')
            return

        ds_list = []
        camera_indices1_list = []
        camera_indices2_list = []
        for image_pair in valid_image_pairs.values():
            # add both directions
            image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
            cam_id1, cam_id2 = image1.cam_id, image2.cam_id
            cam1, cam2 = cameras[cam_id1], cameras[cam_id2]
            principal_point0, principal_point1 = cam1.principal_point, cam2.principal_point
            K0 = np.array([[1, 0, principal_point0[0]], [0, 1, principal_point0[1]], [0, 0, 1]])
            K1 = np.array([[1, 0, principal_point1[0]], [0, 1, principal_point1[1]], [0, 0, 1]])
            i1_G_i0 = K1.T @ image_pair.F @ K0
            ds = fetzer_ds(i1_G_i0)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id1)
            camera_indices2_list.append(cam_id2)
            i0_G_i1 = i1_G_i0.T
            ds = fetzer_ds(i0_G_i1)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id2)
            camera_indices2_list.append(cam_id1)
        
        ds = torch.tensor(np.array(ds_list), dtype=torch.float64).to(self.device).unsqueeze(1)
        camera_indices1 = torch.tensor(np.array(camera_indices1_list), dtype=torch.int64).to(self.device).flatten()
        camera_indices2 = torch.tensor(np.array(camera_indices2_list), dtype=torch.int64).to(self.device).flatten()

        model = FetzerNonBatched(focals)
        strategy = pp.optim.strategy.TrustRegion(radius=1e2, max=1e6, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5) # cuSolverSP()
        cauchy_kernel = Cauchy(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=cauchy_kernel, reject=30)

        input = {
            "ds": ds,
            "camera_indices1": camera_indices1,
            "camera_indices2": camera_indices2
        }

        window_size = 3
        loss_history = []
        progress_bar = tqdm.trange(VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            torch.set_printoptions(threshold=torch.inf)
            print(f'focals: {model.focals}, loss: {loss.item()}')
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.close()

        focals_ = focals.detach().cpu().numpy().squeeze()
        counter = 0
        for cam, focal in zip(cameras, focals_):
            if (focal / np.mean(cam.focal_length) < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio'] or 
                focal / np.mean(cam.focal_length) > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']):
                counter += 1
                continue
            cam.has_refined_focal_length = True
            cam.focal_length = np.array([focal, focal])
        
        print(f'{counter} cameras are rejected in view graph calibration')

        thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

        # manually calculate the residuals
        loss = model.forward(ds, camera_indices1, camera_indices2).detach().cpu().numpy()
        invalid_counter = 0
        loss_sq = np.sum(loss ** 2, axis=-1)
        for idx, (pair_id, image_pair) in enumerate(valid_image_pairs.items()):
            if loss_sq[idx*2] > thres_two_view_error_sq:
                invalid_counter += 1
                view_graph.image_pairs[pair_id].is_valid = False
        print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')