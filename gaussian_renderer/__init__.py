#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, RGB2SH
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

def euler2matrix(yaw):
    cos = torch.cos(-yaw)
    sin = torch.sin(-yaw)
    rot = torch.eye(3).float().cuda()
    rot[0,0] = cos
    rot[0,2] = sin
    rot[2,0] = -sin
    rot[2,2] = cos
    return rot

def cat_bgfg(bg, fg, only_dynamic=False, only_xyz=False):
    if only_xyz:
        bg_feats = [bg.get_xyz]
    else:
        bg_feats = [bg.get_xyz, bg.get_opacity, bg.get_scaling, bg.get_rotation, bg.get_features, bg.get_3D_features]
    
    output = []
    for fg_feat, bg_feat in zip(fg, bg_feats):
        if fg_feat is None:
            output.append(bg_feat)
        elif only_dynamic:
            output.append(fg_feat)
        else:
            output.append(torch.cat((bg_feat, fg_feat), dim=0))
    
    return output


def cat_all_fg(all_fg, next_fg):
    output = []
    for feat, next_feat in zip(all_fg, next_fg):
        if feat is None:
            feat = next_feat
        else:
            feat = torch.cat((feat, next_feat), dim=0)
        output.append(feat)
    return output


def proj_uv(xyz, cam):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intr = torch.as_tensor(cam.K[:3, :3]).float().to(device)  # (3, 3)
    w2c = torch.tensor(cam.w2c).float().to(device)[:3, :]  # (3, 4)

    c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    i_xyz = (intr @ c_xyz.mT).mT  # (N, 3)
    uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3) # (N, 2)
    return uv


def unicycle_b2w(timestamp, model):
    # model = unicycle_models[track_id]['model']
    pred = model(timestamp)
    if pred is None:
        return None
    pred_a, pred_b, pred_v, pred_phi, pred_h = pred
    # r = euler_angles_to_matrix(torch.tensor([0, pred_phi-torch.pi, 0]), 'XYZ')
    rt = torch.eye(4).float().cuda()
    rt[:3,:3] = euler2matrix(pred_phi)
    rt[1, 3], rt[0, 3], rt[2, 3] = pred_h, pred_a, pred_b
    return rt

def render(viewpoint_camera, prev_viewpoint_camera, pc : GaussianModel, dynamic_gaussians : dict, 
                        unicycles : dict, pipe, bg_color : torch.Tensor, 
                        render_optical=False, scaling_modifier = 1.0, only_dynamic=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    timestamp = viewpoint_camera.timestamp

    all_fg = [None, None, None, None, None, None]
    prev_all_fg = [None]

    if len(unicycles) == 0:
        track_dict = viewpoint_camera.dynamics
        if prev_viewpoint_camera is not None:
            prev_track_dict = prev_viewpoint_camera.dynamics
    else:
        track_dict, prev_track_dict = {}, {}
        for track_id, uni_model in unicycles.items():
            B2W = unicycle_b2w(timestamp, uni_model['model'])
            track_dict[track_id] = B2W
            if prev_viewpoint_camera is not None:
                prev_B2W = unicycle_b2w(prev_viewpoint_camera.timestamp, uni_model['model'])
                prev_track_dict[track_id] = prev_B2W

    for track_id, B2W in track_dict.items():
        w_dxyz = (B2W[:3, :3] @ dynamic_gaussians[track_id].get_xyz.T).T + B2W[:3, 3]
        drot = quaternion_to_matrix(dynamic_gaussians[track_id].get_rotation)
        w_drot = matrix_to_quaternion(B2W[:3, :3] @ drot)
        next_fg = [w_dxyz, 
                   dynamic_gaussians[track_id].get_opacity, 
                   dynamic_gaussians[track_id].get_scaling, 
                   w_drot,
                   dynamic_gaussians[track_id].get_features,
                   dynamic_gaussians[track_id].get_3D_features]
        # next_fg = get_next_fg(dynamic_gaussians[track_id], B2W)
        # w_dxyz = next_fg[0]
        all_fg = cat_all_fg(all_fg, next_fg)

        if render_optical and prev_viewpoint_camera is not None:
            if track_id in prev_track_dict:
                prev_B2W = prev_track_dict[track_id]
                prev_w_dxyz = torch.mm(prev_B2W[:3, :3], dynamic_gaussians[track_id].get_xyz.T).T + prev_B2W[:3, 3]
                prev_all_fg = cat_all_fg(prev_all_fg, [prev_w_dxyz])
            else:
                prev_all_fg = cat_all_fg(prev_all_fg, [w_dxyz])
            
    xyz, opacity, scales, rotations, shs, feats3D = cat_bgfg(pc, all_fg)
    if render_optical and prev_viewpoint_camera is not None:
        prev_xyz = cat_bgfg(pc, prev_all_fg, only_xyz=True)[0]
        uv = proj_uv(xyz, viewpoint_camera)
        prev_uv = proj_uv(prev_xyz, prev_viewpoint_camera)
        delta_uv = uv - prev_uv
        delta_uv = torch.cat([delta_uv, torch.ones_like(delta_uv[:, :1], device=delta_uv.device)], dim=-1)
    else:
        delta_uv = torch.zeros_like(xyz)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if pc.affine:
        cam_xyz, cam_dir = viewpoint_camera.c2w[:3, 3].cuda(), viewpoint_camera.c2w[:3, 2].cuda()
        o_enc = pc.pos_enc(cam_xyz[None, :] / 60)
        d_enc = pc.dir_enc(cam_dir[None, :])
        appearance = pc.appearance_model(torch.cat([o_enc, d_enc], dim=1)) * 1e-1
        affine_weight, affine_bias = appearance[:, :9].view(3, 3), appearance[:, -3:]
        affine_weight = affine_weight + torch.eye(3, device=appearance.device)

    # bg_img = pc.sky_model(enc).view(*rays_d.shape).permute(2, 0, 1).float()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = xyz
    means2D = screenspace_points

    cov3D_precomp = None
    colors_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, feats, depth, flow = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        feats3D = feats3D,
        delta = delta_uv)
    
    if pc.affine:
        colors = rendered_image.view(3, -1).permute(1, 0) # (H*W, 3)
        refined_image = (colors @ affine_weight + affine_bias).clip(0, 1).permute(1, 0).view(*rendered_image.shape)
    else:
        refined_image = rendered_image

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": refined_image,
            "feats": feats,
            "depth": depth,
            "opticalflow": flow,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
