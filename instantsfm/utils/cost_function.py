import pyceres
import numpy as np
from scipy.linalg import svd
import torch

from bae.utils.ba import rotate_quat
from bae.autograd.function import TrackingTensor, map_transform

@map_transform
def fetzer_cost(fi, fj, ds):
    di = fj * fj * ds[..., 0, 0] + ds[..., 0, 1]
    dj = fi * fi * ds[..., 2, 0] + ds[..., 2, 2]
    di = torch.where(di == 0, torch.tensor(1e-6, dtype=torch.float64, device=di.device), di)
    dj = torch.where(dj == 0, torch.tensor(1e-6, dtype=torch.float64, device=dj.device), dj)

    K0_01 = -(fj * fj * ds[..., 0, 2] + ds[..., 0, 3]) / di
    K1_12 = -(fi * fi * ds[..., 2, 1] + ds[..., 2, 3]) / dj

    loss = torch.cat([(fi * fi - K0_01) / (fi * fi), (fj * fj - K1_12) / (fj * fj)], dim=-1) # (num_pairs, 2)
    return loss

@map_transform
def pairwise_cost(points, camera_translations, scales, translations, is_calibrated):
    positions1 = camera_translations
    positions2 = points
    loss = translations - scales * (positions2 - positions1)
    calibrated_factor = torch.where(is_calibrated, 1.0, 0.5).unsqueeze(-1)
    loss = loss * calibrated_factor
    return loss

# functions that reproject points from cameras to images, each function is based on a different camera model
@map_transform
def reproject_simple_pinhole(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -1].unsqueeze(-1)
    points_proj = points_proj * f + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_pinhole(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -2:]
    points_proj = points_proj * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_simple_radial(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -2].unsqueeze(-1)
    k = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    points_proj = points_proj * (1 + k * r2) * f + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_radial(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -3].unsqueeze(-1)
    k1 = intrinsics[..., -2].unsqueeze(-1)
    k2 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    points_proj = points_proj * (1 + k1 * r2 + k2 * r2**2) * f + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_opencv(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -6:-4]
    k1 = intrinsics[..., -4].unsqueeze(-1)
    k2 = intrinsics[..., -3].unsqueeze(-1)
    p = intrinsics[..., -2:]
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    uv = (points_proj[..., 0] * points_proj[..., 1]).unsqueeze(-1)
    radial = k1 * r2 + k2 * r2**2
    d = points_proj * radial + 2 * p * uv
    d = d + p.flip(-1) * (r2 + 2 * points_proj[..., :2]**2)
    points_proj = points_proj + d
    points_proj = points_proj * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_opencv_fisheye(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -6:-4]
    k1 = intrinsics[..., -4].unsqueeze(-1)
    k2 = intrinsics[..., -3].unsqueeze(-1)
    k3 = intrinsics[..., -2].unsqueeze(-1)
    # k4 = intrinsics[..., -1].unsqueeze(-1) but ignored
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    points_proj = points_proj * radial
    points_proj = points_proj * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_full_opencv(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -10:-8]
    k1 = intrinsics[..., -8].unsqueeze(-1)
    k2 = intrinsics[..., -7].unsqueeze(-1)
    p = intrinsics[..., -6:-4]
    k3 = intrinsics[..., -4].unsqueeze(-1)
    k4 = intrinsics[..., -3].unsqueeze(-1)
    k5 = intrinsics[..., -2].unsqueeze(-1)
    k6 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    uv = (points_proj[..., 0] * points_proj[..., 1]).unsqueeze(-1)
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3) - 1
    d = points_proj * radial + 2 * p * uv
    d = d + p.flip(-1) * (r2 + 2 * points_proj[..., :2]**2)
    points_proj = points_proj + d
    points_proj = points_proj * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_fov(points, extrinsics, intrinsics, pp):
    # TODO: complete this function
    raise NotImplementedError
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)  # add dimension for broadcasting
    ff = intrinsics[..., -2].unsqueeze(-1)
    omega = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    omega2 = omega * omega
    epsilon = 1e-4
    factor = torch.zeros_like(omega)
    # omega close to 0
    omega_mask = omega2 < epsilon
    factor[omega_mask] = (omega2[omega_mask] + r2[omega_mask]) / 3 - omega2[omega_mask] / 12 + 1
    # r close to 0
    r_mask = (r2 < epsilon) & ~omega_mask
    factor[r_mask] = (-2 * torch.tan(omega[r_mask] / 2) * (4 * r2[r_mask] * torch.tan(omega[r_mask] / 2)**2 - 3)) / (3 * omega[r_mask])
    # else
    else_mask = ~omega_mask & ~r_mask
    radius = torch.sqrt(r2[else_mask])
    numerator = torch.atan(radius * 2 * torch.tan(omega[else_mask] / 2))
    factor[else_mask] = numerator / (radius * omega[else_mask])

    points_proj = points_proj * factor * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_simple_radial_fisheye(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -2].unsqueeze(-1)
    k = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    points_proj = points_proj * (1 + k * r2) * f + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_radial_fisheye(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -3].unsqueeze(-1)
    k1 = intrinsics[..., -2].unsqueeze(-1)
    k2 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    points_proj = points_proj * (1 + k1 * r2 + k2 * r2**2) * f + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

@map_transform
def reproject_thin_prism_fisheye(points, extrinsics, intrinsics, pp):
    # TODO: complete this function
    raise NotImplementedError
    points_cam = rotate_quat(points, extrinsics)
    depth = points_cam[..., 2].unsqueeze(-1)  # [N, 1]
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -10:-8]
    k1 = intrinsics[..., -8].unsqueeze(-1)
    k2 = intrinsics[..., -7].unsqueeze(-1)
    p = intrinsics[..., -6:-4]
    k3 = intrinsics[..., -4].unsqueeze(-1)
    # k4 = intrinsics[..., -3].unsqueeze(-1) but ignored
    sx = intrinsics[..., -2:]
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    uv = (points_proj[..., 0] * points_proj[..., 1]).unsqueeze(-1)
    radial = k1 * r2 + k2 * r2**2 + k3 * r2**3
    d = points_proj * radial + 2 * p * uv
    d = d + p.flip(-1) * (r2 + 2 * points_proj[..., :2]**2)
    d = d + sx * r2
    points_proj = points_proj + d
    points_proj = points_proj * ff + pp
    # Return [N, 3] with (x, y, depth)
    return torch.cat([points_proj, depth], dim=-1)

# No-depth versions: return only [N, 2] (x, y) without depth
@map_transform
def reproject_simple_pinhole_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -1].unsqueeze(-1)
    points_proj = points_proj * f + pp
    return points_proj

@map_transform
def reproject_pinhole_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -2:]
    points_proj = points_proj * ff + pp
    return points_proj

@map_transform
def reproject_simple_radial_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -2].unsqueeze(-1)
    k = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    points_proj = points_proj * (1 + k * r2) * f + pp
    return points_proj

@map_transform
def reproject_radial_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -3].unsqueeze(-1)
    k1 = intrinsics[..., -2].unsqueeze(-1)
    k2 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    points_proj = points_proj * (1 + k1 * r2 + k2 * r2**2) * f + pp
    return points_proj

@map_transform
def reproject_opencv_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -6:-4]
    k1 = intrinsics[..., -4].unsqueeze(-1)
    k2 = intrinsics[..., -3].unsqueeze(-1)
    p = intrinsics[..., -2:]
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    uv = (points_proj[..., 0] * points_proj[..., 1]).unsqueeze(-1)
    radial = k1 * r2 + k2 * r2**2
    d = points_proj * radial + 2 * p * uv
    d = d + p.flip(-1) * (r2 + 2 * points_proj[..., :2]**2)
    points_proj = points_proj + d
    points_proj = points_proj * ff + pp
    return points_proj

@map_transform
def reproject_opencv_fisheye_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -6:-4]
    k1 = intrinsics[..., -4].unsqueeze(-1)
    k2 = intrinsics[..., -3].unsqueeze(-1)
    k3 = intrinsics[..., -2].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    points_proj = points_proj * radial
    points_proj = points_proj * ff + pp
    return points_proj

@map_transform
def reproject_full_opencv_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    ff = intrinsics[..., -10:-8]
    k1 = intrinsics[..., -8].unsqueeze(-1)
    k2 = intrinsics[..., -7].unsqueeze(-1)
    p = intrinsics[..., -6:-4]
    k3 = intrinsics[..., -4].unsqueeze(-1)
    k4 = intrinsics[..., -3].unsqueeze(-1)
    k5 = intrinsics[..., -2].unsqueeze(-1)
    k6 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    uv = (points_proj[..., 0] * points_proj[..., 1]).unsqueeze(-1)
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3) - 1
    d = points_proj * radial + 2 * p * uv
    d = d + p.flip(-1) * (r2 + 2 * points_proj[..., :2]**2)
    points_proj = points_proj + d
    points_proj = points_proj * ff + pp
    return points_proj

@map_transform
def reproject_fov_no_depth(points, extrinsics, intrinsics, pp):
    raise NotImplementedError

@map_transform
def reproject_simple_radial_fisheye_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -2].unsqueeze(-1)
    k = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    points_proj = points_proj * (1 + k * r2) * f + pp
    return points_proj

@map_transform
def reproject_radial_fisheye_no_depth(points, extrinsics, intrinsics, pp):
    points_cam = rotate_quat(points, extrinsics)
    points_proj = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
    f = intrinsics[..., -3].unsqueeze(-1)
    k1 = intrinsics[..., -2].unsqueeze(-1)
    k2 = intrinsics[..., -1].unsqueeze(-1)
    r2 = torch.sum(points_proj[..., :2]**2, dim=-1).unsqueeze(-1)
    r = torch.sqrt(r2)
    theta = torch.atan(r)
    points_proj = points_proj * theta / r
    points_proj = points_proj * (1 + k1 * r2 + k2 * r2**2) * f + pp
    return points_proj

@map_transform
def reproject_thin_prism_fisheye_no_depth(points, extrinsics, intrinsics, pp):
    raise NotImplementedError

# all the reprojection functions are based on the camera model used, import a list of functions can simplify the code
reproject_funcs = [reproject_simple_pinhole, reproject_pinhole, reproject_simple_radial, reproject_radial, reproject_opencv,
                   reproject_opencv_fisheye, reproject_full_opencv, reproject_fov, reproject_simple_radial_fisheye,
                   reproject_radial_fisheye, reproject_thin_prism_fisheye]

reproject_funcs_no_depth = [reproject_simple_pinhole_no_depth, reproject_pinhole_no_depth, reproject_simple_radial_no_depth, 
                             reproject_radial_no_depth, reproject_opencv_no_depth, reproject_opencv_fisheye_no_depth, 
                             reproject_full_opencv_no_depth, reproject_fov_no_depth, reproject_simple_radial_fisheye_no_depth,
                             reproject_radial_fisheye_no_depth, reproject_thin_prism_fisheye_no_depth]

def fetzer_d(ai, bi, aj, bj, u, v):
    d = np.zeros(4)
    d[0] = ai[u] * aj[v] - ai[v] * aj[u]
    d[1] = ai[u] * bj[v] - ai[v] * bj[u]
    d[2] = bi[u] * aj[v] - bi[v] * aj[u]
    d[3] = bi[u] * bj[v] - bi[v] * bj[u]
    return d

def fetzer_ds(i1_G_i0):
    U, s, Vt = svd(i1_G_i0)
    V = Vt.T

    v_0 = V[:, 0]
    v_1 = V[:, 1]

    u_0 = U[:, 0]
    u_1 = U[:, 1]

    ai = np.array([
        s[0] * s[0] * (v_0[0] * v_0[0] + v_0[1] * v_0[1]),
        s[0] * s[1] * (v_0[0] * v_1[0] + v_0[1] * v_1[1]),
        s[1] * s[1] * (v_1[0] * v_1[0] + v_1[1] * v_1[1])
    ])

    aj = np.array([
        u_1[0] * u_1[0] + u_1[1] * u_1[1],
        -(u_0[0] * u_1[0] + u_0[1] * u_1[1]),
        u_0[0] * u_0[0] + u_0[1] * u_0[1]
    ])

    bi = np.array([
        s[0] * s[0] * v_0[2] * v_0[2],
        s[0] * s[1] * v_0[2] * v_1[2],
        s[1] * s[1] * v_1[2] * v_1[2]
    ])

    bj = np.array([
        u_1[2] * u_1[2],
        -(u_0[2] * u_1[2]),
        u_0[2] * u_0[2]
    ])

    d_01 = fetzer_d(ai, bi, aj, bj, 1, 0)
    d_02 = fetzer_d(ai, bi, aj, bj, 0, 2)
    d_12 = fetzer_d(ai, bi, aj, bj, 2, 1)

    ds = [d_01, d_02, d_12]

    return ds

class FetzerFocalLengthCostFunction(pyceres.CostFunction):
    def __init__(self, i1_f_i0, principal_point0, principal_point1):
        pyceres.CostFunction.__init__(self)
        K0 = np.array([[1, 0, principal_point0[0]], [0, 1, principal_point0[1]], [0, 0, 1]])
        K1 = np.array([[1, 0, principal_point1[0]], [0, 1, principal_point1[1]], [0, 0, 1]])
        i1_G_i0 = K1.T @ i1_f_i0 @ K0
        self.ds = fetzer_ds(i1_G_i0)
        self.set_parameter_block_sizes([1, 1])
        self.set_num_residuals(2)
    
    def Evaluate(self, parameters, residuals, jacobians):
        d_01 = self.ds[0]
        d_12 = self.ds[2]
        fi = parameters[0][0]
        fj = parameters[1][0]
        
        di = fj * fj * d_01[0] + d_01[1]
        dj = fi * fi * d_12[0] + d_12[2]
        di = 1e-6 if di == 0 else di
        dj = 1e-6 if dj == 0 else dj

        K0_01 = -(fj * fj * d_01[2] + d_01[3]) / di
        K1_12 = -(fi * fi * d_12[1] + d_12[3]) / dj
        residuals[0] = (fi * fi - K0_01) / (fi * fi)
        residuals[1] = (fj * fj - K1_12) / (fj * fj)

        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = np.array([2 * K0_01 / (fi * fi * fi), 2 * fj * (d_01[2] + d_01[0] * K0_01) / (fi * fi * di)])
            if jacobians[1] is not None:
                jacobians[1][:] = np.array([2 * fi * (d_12[1] + d_12[0] * K1_12) / (fj * fj * dj), 2 * K1_12 / (fj * fj * fj)])
        
        return True
    
class FetzerFocalLengthSameCameraCostFunction(pyceres.CostFunction):
    def __init__(self, i1_f_i0, principal_point):
        pyceres.CostFunction.__init__(self)
        K0 = np.array([[1, 0, principal_point[0]], [0, 1, principal_point[1]], [0, 0, 1]])
        K1 = np.array([[1, 0, principal_point[0]], [0, 1, principal_point[1]], [0, 0, 1]])
        i1_G_i0 = K1.T @ i1_f_i0 @ K0
        self.ds = fetzer_ds(i1_G_i0)
        self.set_parameter_block_sizes([1])
        self.set_num_residuals(2)
    
    def Evaluate(self, parameters, residuals, jacobians):
        d_01 = self.ds[0]
        d_12 = self.ds[2]
        fi = parameters[0][0]
        fj = fi  # Same camera, so fi and fj are the same
        
        di = fj * fj * d_01[0] + d_01[1]
        dj = fi * fi * d_12[0] + d_12[2]
        
        EPS = 1e-6
        K0_01 = -(fj * fj * d_01[2] + d_01[3]) / (di + EPS)
        K1_12 = -(fi * fi * d_12[1] + d_12[3]) / (dj + EPS)
        residuals[0] = (fi * fi - K0_01) / (fi * fi)
        residuals[1] = (fj * fj - K1_12) / (fj * fj)

        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = np.array([2 * (d_01[1] * d_01[3] + 2 * d_01[0] * d_01[3] * fi**2 + d_01[0] * d_01[2] * fi**4) / (fi**3 * (d_01[1] + d_01[0] * fi**2)**2),
                                            2 * (d_12[2] * d_12[3] + 2 * d_12[0] * d_12[3] * fi**2 + d_12[0] * d_12[1] * fi**4) / (fi**3 * (d_12[2] + d_12[0] * fi**2)**2)])
        
        return True