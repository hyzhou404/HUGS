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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from copy import deepcopy
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from utils.semantic_utils import colorize
import flow_vis_torch
from utils.cmap import color_depth_map
from imageio.v2 import imwrite

def to4x4(R, T):
    RT = np.eye(4,4)
    RT[:3, :3] = R
    RT[:3, 3] = T
    return RT

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[0, ...]].permute(2, 0, 1)


def apply_depth_colormap(depth, near_plane=None, far_plane=None, cmap="turbo"):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image


def render_set(model_path, name, iteration, views, scene, pipeline, background, data_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
    optical_path = os.path.join(model_path, name, "ours_{}".format(iteration), "optical")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "error_map")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(semantic_path, exist_ok=True)
    makedirs(optical_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        if data_type == 'kitti':
            gap = 2
        elif data_type == 'kitti360':
            gap = 4
        elif data_type == 'waymo':
            gap = 1
        elif data_type == 'nuscenes' or data_type == 'pandaset':
            gap = 6

        if idx - gap < 0:
            prev_view = None
        else:
            prev_view = views[idx-4]
        render_pkg = render(
            view, prev_view, scene.gaussians, scene.dynamic_gaussians, scene.unicycles, pipeline, background, True
        )
        rendering = render_pkg['render'].detach().cpu()
        semantic = render_pkg['feats'].detach().cpu()
        semantic = torch.argmax(semantic, dim=0)
        semantic_rgb = colorize(semantic.detach().cpu().numpy())
        depth = render_pkg['depth']
        color_depth = color_depth_map(depth[0].detach().cpu().numpy())
        color_depth[semantic == 10] = np.array([255.0, 255.0, 255.0])
        gt = view.original_image[0:3, :, :]
        
        # _, ssim_map = ssim(rendering[None, ...], gt[None, ...], return_full_image=True)
        # ssim_map = torch.mean(ssim_map[0], dim=0).clip(0, 1)[None, ...]
        # error_map = 1 - ssim_maps
        error_map = torch.mean((rendering - gt) ** 2, dim=0)[None, ...]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(1.408, 0.376)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow((error_map.detach().cpu().numpy().transpose(1,2,0)), cmap='jet')
        plt.savefig(os.path.join(error_path, view.image_name + ".png"), dpi=1000)
        plt.close('all')

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        semantic_rgb.save(os.path.join(semantic_path, view.image_name + ".png"))
        imwrite(os.path.join(depth_path, view.image_name + ".png"), color_depth)
        
        opticalflow = render_pkg["opticalflow"]
        opticalflow = opticalflow.permute(1,2,0)
        opticalflow = opticalflow[..., :2]
        pytorch_optic_rgb = flow_vis_torch.flow_to_color(opticalflow.permute(2, 0, 1))  # (2, h, w)
        torchvision.utils.save_image(pytorch_optic_rgb.float(), os.path.join(optical_path, view.image_name + ".png"), normalize=True)
        # torchvision.utils.save_image(error_map, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, data_type, affine, ignore_dynamic):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, affine=affine)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, data_type=data_type, ignore_dynamic=ignore_dynamic)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, background, data_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background, data_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--data_type", default='kitti360', type=str)
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--ignore_dynamic", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.source_path = os.path.join(args.model_path, 'data')

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.data_type, args.affine, args.ignore_dynamic)