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

import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch.nn.functional as F
from imageio.v2 import imread
import torch
import random


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx_ratio: float
    cy_ratio: float
    semantic2d: np.array
    optical_image: np.array
    mask: np.array
    timestamp: int
    dynamics: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    verts: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4]) # cam_centers in world coordinate

    center, diagonal = get_center_and_diag(cam_centers)
    # radius = diagonal * 1.1 + 30
    radius = 10

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        print('Create random colors')
        # shs = np.random.random((positions.shape[0], 3)) / 255.0
        shs = np.ones((positions.shape[0], 3)) * 0.5
        colors = SH2RGB(shs)
    # shs = np.ones((positions.shape[0], 3)) * 0.5
    # colors = SH2RGB(shs)
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readStudioCameras(path, white_background, data_type, ignore_dynamic):
    train_cam_infos, test_cam_infos = [], []
    with open(os.path.join(path, 'meta_data.json')) as json_file:
        meta_data = json.load(json_file)

        verts = {}
        if 'verts' in meta_data and not ignore_dynamic:
            verts_list = meta_data['verts']
            for k, v in verts_list.items():
                verts[k] = np.array(v)

        frames = meta_data['frames']
        for idx, frame in enumerate(frames):
            matrix = np.linalg.inv(np.array(frame['camtoworld']))
            R = matrix[:3, :3]
            T = matrix[:3, 3]
            R = np.transpose(R)

            rgb_path = os.path.join(path, frame['rgb_path'].replace('./', ''))

            rgb_split = rgb_path.split('/')
            image_name = '_'.join([rgb_split[-2], rgb_split[-1][:-4]])
            image = Image.open(rgb_path)

            semantic_2d = None
            semantic_pth = rgb_path.replace("images", "semantics").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(semantic_pth):
                semantic_2d = np.load(semantic_pth)
                semantic_2d[(semantic_2d == 14) | (semantic_2d == 15)] = 13

            optical_path = rgb_path.replace("images", "flow").replace('.png', '_flow.npy').replace('.jpg', '_flow.npy')
            if os.path.exists(optical_path):
                optical_image = np.load(optical_path)
            else:
                optical_image = None

            mask = None
            mask_path = rgb_path.replace("images", "masks").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

            timestamp = frame.get('timestamp', -1)

            intrinsic = np.array(frame['intrinsics'])
            FovX = focal2fov(intrinsic[0, 0], image.size[0])
            FovY = focal2fov(intrinsic[1, 1], image.size[1])
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            w, h = image.size
            
            dynamics = {}
            if 'dynamics' in frame and not ignore_dynamic:
                dynamics_list = frame['dynamics']
                for iid in dynamics_list.keys():
                    dynamics[iid] = torch.tensor(dynamics_list[iid]).cuda()
                
            cam_info = CameraInfo(uid=idx, R=R, T=T, K=intrinsic, FovY=FovY, FovX=FovX, image=image,
                                image_path=rgb_path, image_name=image_name, width=image.size[0],
                                height=image.size[1], cx_ratio=2*cx/w, cy_ratio=2*cy/h, semantic2d=semantic_2d, 
                                optical_image=optical_image, mask=mask, timestamp=timestamp, dynamics=dynamics)
            
            # kitti360
            if data_type == 'kitti360':
                # if 'cam_2' in cam_info.image_name or 'cam_3' in cam_info.image_name:
                #     train_cam_infos.append(cam_info)
                #     # continue
                if idx < 20:
                    train_cam_infos.append(cam_info)
                elif idx % 8 < 4:
                    train_cam_infos.append(cam_info)
                elif idx % 8 >= 4:
                    test_cam_infos.append(cam_info)
                else:
                    continue

            elif data_type == 'kitti':
                if idx < 10 or idx >= len(frames) - 4:
                    train_cam_infos.append(cam_info)
                elif idx % 4 < 2:
                    train_cam_infos.append(cam_info)
                elif idx % 4 == 2:
                    test_cam_infos.append(cam_info)
                else:
                    continue

            elif data_type == "nuscenes":
                if idx < 600 or idx >= 1200:
                    continue
                elif idx % 30 >= 24:
                    # print('test', cam_info.image_name)
                    test_cam_infos.append(cam_info)
                else:
                    # print('train', cam_info.image_name)
                    train_cam_infos.append(cam_info)

            elif data_type == "waymo":
                if idx > 10 and idx % 10 >= 9:
                    test_cam_infos.append(cam_info)
                else:
                    train_cam_infos.append(cam_info)

            elif data_type == "pandaset":
                # if idx >= 360:
                #     continue
                if idx > 30 and idx % 30 >= 24:
                    test_cam_infos.append(cam_info)
                else:
                    train_cam_infos.append(cam_info)
            
            else:
                raise NotImplementedError
    return train_cam_infos, test_cam_infos, verts


def readStudioInfo(path, white_background, eval, data_type, ignore_dynamic):
    train_cam_infos, test_cam_infos, verts = readStudioCameras(path, white_background, data_type, ignore_dynamic)

    print(f'Loaded {len(train_cam_infos)} train cameras and {len(test_cam_infos)} test cameras')
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, 'lidar', 'cat.ply')
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 500_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        AABB = [[-20, -25, -20], [20, 5, 80]]
        xyz = np.random.uniform(AABB[0], AABB[1], (500000, 3))
        # xyz = np.load(os.path.join(path, 'lidar_point.npy'))
        num_pts = xyz.shape[0]
        shs = np.ones((num_pts, 3)) * 0.5
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print('When loading point clound, meet error:', e)
        exit(0)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           verts=verts)
    return scene_info


sceneLoadTypeCallbacks = {
    "Studio": readStudioInfo,
}