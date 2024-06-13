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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import open3d as o3d
import numpy as np
from utils.dynamic_utils import create_unicycle_model
import shutil

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, 
                 unicycle=False, uc_fit_iter=0, resolution_scales=[1.0], data_type='kitti360', ignore_dynamic=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "ckpts"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            raise NotImplementedError
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            # scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            raise NotImplementedError
        elif os.path.exists(os.path.join(args.source_path, "meta_data.json")):
            print("Found meta_data.json file, assuming Studio data set!")
            scene_info = sceneLoadTypeCallbacks['Studio'](args.source_path, args.white_background, args.eval, data_type, ignore_dynamic)
        else:
            assert False, "Could not recognize scene type!"

        self.dynamic_verts = scene_info.verts
        self.dynamic_gaussians = {}
        for track_id in scene_info.verts:
            self.dynamic_gaussians[track_id] = GaussianModel(args.sh_degree, feat_mutable=False)
        
        if unicycle:
            self.unicycles = create_unicycle_model(scene_info.train_cameras, self.model_path, uc_fit_iter, data_type)
        else:
            self.unicycles = {}

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            shutil.copyfile(os.path.join(args.source_path, 'meta_data.json'), os.path.join(self.model_path, 'meta_data.json'))

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            (model_params, first_iter) = torch.load(os.path.join(self.model_path, "ckpts", f"chkpnt{self.loaded_iter}.pth"))
            gaussians.restore(model_params, None)
            for iid, dynamic_gaussian in self.dynamic_gaussians.items():
                (model_params, first_iter) = torch.load(os.path.join(self.model_path, "ckpts", f"dynamic_{iid}_chkpnt{self.loaded_iter}.pth"))
                dynamic_gaussian.restore(model_params, None)
            for iid, unicycle_pkg in self.unicycles.items():
                model_params = torch.load(os.path.join(self.model_path, "ckpts", f"unicycle_{iid}_chkpnt{self.loaded_iter}.pth"))
                unicycle_pkg['model'].restore(model_params)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            for track_id in self.dynamic_gaussians.keys():
                vertices = scene_info.verts[track_id]

                # init from template
                l, h, w = vertices[:, 0].max() - vertices[:, 0].min(), vertices[:, 1].max() - vertices[:, 1].min(), vertices[:, 2].max() - vertices[:, 2].min()
                pcd = o3d.io.read_point_cloud(f"utils/vehicle_template/benz_{data_type}.ply")
                points = np.array(pcd.points) * np.array([l, h, w])
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) * 0.5)

                self.dynamic_gaussians[track_id].create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        point_cloud_vis_path = os.path.join(self.model_path, "point_cloud_vis/iteration_{}".format(iteration))
        self.gaussians.save_vis_ply(os.path.join(point_cloud_vis_path, "point.ply"))
        for iid, dynamic_gaussian in self.dynamic_gaussians.items():
            dynamic_gaussian.save_vis_ply(os.path.join(point_cloud_vis_path, f"dynamic_{iid}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]