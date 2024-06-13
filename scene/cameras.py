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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import decode_op

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", 
                 cx_ratio=None, cy_ratio=None, semantic2d=None, mask=None, timestamp=-1, optical_image=None, dynamics={}
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.cx_ratio = cx_ratio
        self.cy_ratio = cy_ratio
        self.timestamp = timestamp
        _, self.H, self.W = image.shape
        self.w2c = np.eye(4)
        self.w2c[:3, :3] = self.R.T
        self.w2c[:3, 3] = self.T
        self.c2w = torch.from_numpy(np.linalg.inv(self.w2c)).cuda()
        self.fx = fov2focal(self.FoVx, self.W)
        self.fy = fov2focal(self.FoVy, self.H)
        self.dynamics = dynamics

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if semantic2d is not None:
            self.semantic2d = semantic2d.to(self.data_device)
        else:
            self.semantic2d = None
        if mask is not None:
            self.mask = torch.from_numpy(mask).bool().to(self.data_device)
        else:
            self.mask = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if optical_image is not None:
            self.optical_gt = torch.from_numpy(optical_image).to(self.data_device)
        else:
            self.optical_gt = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                     fovX=self.FoVx, fovY=self.FoVy, cx_ratio=cx_ratio, cy_ratio=cy_ratio).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_rays(self):
        i, j = torch.meshgrid(torch.linspace(0, self.W-1, self.W), 
                              torch.linspace(0, self.H-1, self.H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-self.cx_ratio)/self.fx, -(j-self.cy_ratio)/self.fy, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * self.c2w[:3,:3], -1).to(self.data_device)
        rays_o = self.c2w[:3,-1].expand(rays_d.shape).to(self.data_device)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        return rays_o.permute(2,0,1), rays_d.permute(2,0,1)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

