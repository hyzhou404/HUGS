import numpy as np
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
import os


def rot2Euler(R):
    sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2,1] , R[2,2])
        y = torch.atan2(-R[2,0], sy)
        z = torch.atan2(R[1,0], R[0,0])
    else:
        x = torch.atan2(-R[1,2], R[1,1])
        y = torch.atan2(-R[2,0], sy)
        z = 0

    return torch.stack([x,y,z])

class unicycle(torch.nn.Module):

    def __init__(self, train_timestamp, centers=None, heights=None, phis=None):
        super(unicycle, self).__init__()
        self.train_timestamp = train_timestamp
        self.delta = torch.diff(self.train_timestamp)

        self.input_a = centers[:, 0].clone()
        self.input_b = centers[:, 1].clone()

        if centers is None:
            self.a = nn.Parameter(torch.zeros_like(train_timestamp).float())
            self.b = nn.Parameter(torch.zeros_like(train_timestamp).float())
        else:
            self.a = nn.Parameter(centers[:, 0])
            self.b = nn.Parameter(centers[:, 1])
        
        diff_a = torch.diff(centers[:, 0]) / self.delta
        diff_b = torch.diff(centers[:, 1]) / self.delta
        v = torch.sqrt(diff_a ** 2 + diff_b**2)
        self.v = nn.Parameter(F.pad(v, (0, 1), 'constant', v[-1].item()))
        self.phi = nn.Parameter(phis)

        if heights is None:
            self.h = nn.Parameter(torch.zeros_like(train_timestamp).float())
        else:
            self.h = nn.Parameter(heights)

    def acc_omega(self):
        acc = torch.diff(self.v) / self.delta
        omega = torch.diff(self.phi) / self.delta
        acc = F.pad(acc, (0, 1), 'constant', acc[-1].item())
        omega = F.pad(omega, (0, 1), 'constant', omega[-1].item())
        return acc, omega

    def forward(self, timestamps):
        idx = torch.searchsorted(self.train_timestamp, timestamps, side='left')
        invalid = (idx == self.train_timestamp.shape[0])
        idx[invalid] -= 1
        idx[self.train_timestamp[idx] != timestamps] -= 1
        idx[invalid] += 1
        prev_timestamps = self.train_timestamp[idx]
        delta_t = timestamps - prev_timestamps
        prev_a, prev_b = self.a[idx], self.b[idx]
        prev_v, prev_phi = self.v[idx], self.phi[idx]
        
        acc, omega = self.acc_omega()
        v = prev_v + acc[idx] * delta_t
        phi = prev_phi + omega[idx] * delta_t
        a = prev_a + prev_v * ((torch.sin(phi) - torch.sin(prev_phi)) / (omega[idx] + 1e-6))
        b = prev_b - prev_v * ((torch.cos(phi) - torch.cos(prev_phi)) / (omega[idx] + 1e-6))
        h = self.h[idx]
        return a, b, v, phi, h

    def capture(self):
        return (
            self.a,
            self.b,
            self.v,
            self.phi,
            self.h,
            self.train_timestamp,
            self.delta
        )
    
    def restore(self, model_args):
        (
            self.a,
            self.b,
            self.v,
            self.phi,
            self.h,
            self.train_timestamp,
            self.delta
        ) = model_args

    def visualize(self, save_path, noise_centers=None, gt_centers=None):
        a, b, _, phi, _ = self.forward(self.train_timestamp)
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        phi = phi.detach().cpu().numpy()
        plt.scatter(a, b, marker='x', color='b')
        plt.quiver(a, b, np.ones_like(a) * np.cos(phi), np.ones_like(b) * np.sin(phi), scale=20, width=0.005)
        if noise_centers is not None:
            noise_centers = noise_centers.detach().cpu().numpy()
            plt.scatter(noise_centers[:, 0], noise_centers[:, 1], marker='o', color='gray')
        if gt_centers is not None:
            gt_centers = gt_centers.detach().cpu().numpy()
            plt.scatter(gt_centers[:, 0], gt_centers[:, 1], marker='v', color='g')
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()

    def reg_loss(self):
        reg = 0
        acc, omega = self.acc_omega()
        reg += torch.mean(torch.abs(torch.diff(acc))) * 1
        reg += torch.mean(torch.abs(torch.diff(omega))) * 1
        reg_a_motion = self.v[:-1] * ((torch.sin(self.phi[1:]) - torch.sin(self.phi[:-1])) / (omega[:-1] + 1e-6)) 
        reg_b_motion = -self.v[:-1] * ((torch.cos(self.phi[1:]) - torch.cos(self.phi[:-1])) / (omega[:-1] + 1e-6))
        reg_a = self.a[:-1] + reg_a_motion
        reg_b = self.b[:-1] + reg_b_motion
        reg += torch.mean((reg_a - self.a[1:])**2 + (reg_b - self.b[1:])**2) * 1
        return reg
    
    def pos_loss(self):
        # a, b, _, _, _ = self.forward(self.train_timestamp)
        return torch.mean((self.a - self.input_a) ** 2 + (self.b - self.input_b) ** 2) * 10
    

def create_unicycle_model(train_cams, model_path, opt_iter=0, data_type='kitti'):
    unicycle_models = {}
    if data_type == 'kitti':
        cameras = [cam for cam in train_cams if 'cam_0' in cam.image_name]
    elif data_type == 'waymo':
        cameras = [cam for cam in train_cams if 'cam_1' in cam.image_name]
    else:
        raise NotImplementedError    

    all_centers, all_heights, all_phis, all_timestamps = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    seq_timestamps = []
    for cam in cameras:
        t = cam.timestamp
        seq_timestamps.append(t)
        for track_id, b2w in cam.dynamics.items():
            all_centers[track_id].append(b2w[[0, 2], 3])
            all_heights[track_id].append(b2w[1, 3])
            eulers = rot2Euler(b2w[:3, :3])
            all_phis[track_id].append(eulers[1])
            all_timestamps[track_id].append(t)

    for track_id in all_centers.keys():
        centers = torch.stack(all_centers[track_id], dim=0).cuda()
        timestamps = torch.tensor(all_timestamps[track_id]).cuda()
        heights = torch.tensor(all_heights[track_id]).cuda()
        phis = torch.tensor(all_phis[track_id]).cuda() + torch.pi
        model = unicycle(timestamps, centers.clone(), heights.clone(), phis.clone())
        l = [
            {'params': [model.a], 'lr': 1e-2, "name": "a"},
            {'params': [model.b], 'lr': 1e-2, "name": "b"},
            {'params': [model.v], 'lr': 1e-3, "name": "v"},
            {'params': [model.phi], 'lr': 1e-4, "name": "phi"},
            {'params': [model.h], 'lr': 0, "name": "h"}
        ]

        optimizer = optim.Adam(l, lr=0.0)

        t_range = tqdm(range(opt_iter), desc=f"Fitting {track_id}")
        for iter in t_range:
            loss = 0.2 * model.pos_loss() + model.reg_loss()
            t_range.set_postfix({'loss': loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        unicycle_models[track_id] = {'model': model, 
                                    'optimizer': optimizer,
                                    'input_centers': centers}
    
    os.makedirs(os.path.join(model_path, "unicycle"), exist_ok=True)
    for track_id, unicycle_pkg in unicycle_models.items():
        model = unicycle_pkg['model']
        optimizer = unicycle_pkg['optimizer']
        
        model.visualize(os.path.join(model_path, "unicycle", f"{track_id}_init.png"),
                        # noise_centers=unicycle_pkg['input_centers']
                        )
                        # gt_centers=gt_centers)

    return unicycle_models