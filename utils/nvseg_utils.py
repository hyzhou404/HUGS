import sys
sys.path.append("/data0/hyzhou/workspace/nv_seg")
from network import get_model
from config import cfg, torch_version_float
from datasets.cityscapes import Loader as dataset_cls
from runx.logx import logx
import cv2
import torch
from imageio.v2 import imread, imwrite
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from torchvision.utils import save_image

def restore_net(net, checkpoint):
    assert 'state_dict' in checkpoint, 'cant find state_dict in checkpoint'
    forgiving_state_restore(net, checkpoint['state_dict'])


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        else:            
            logx.msg("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def get_nvseg_model():
    logx.initialize(logdir="./results",
                    global_rank=0)

    cfg.immutable(False)
    cfg.DATASET.NUM_CLASSES = dataset_cls.num_classes
    cfg.DATASET.IGNORE_LABEL = dataset_cls.ignore_label
    cfg.MODEL.MSCALE = True
    cfg.MODEL.N_SCALES = [0.5,1.0,2.0]
    cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
    cfg.OPTIONS.TORCH_VERSION = torch_version_float()
    cfg.DATASET_INST = dataset_cls('folder')
    cfg.immutable(True)
    colorize_mask_fn = cfg.DATASET_INST.colorize_mask

    net = get_model(network='network.ocrnet.HRNet_Mscale',
                    num_classes=cfg.DATASET.NUM_CLASSES,
                    criterion=None)

    snapshot = "ASSETS_PATH/seg_weights/cityscapes_trainval_ocr.HRNet_Mscale_nimble-chihuahua.pth".replace('ASSETS_PATH', cfg.ASSETS_PATH)
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    renamed_ckpt = {'state_dict': {}}
    for k, v in checkpoint['state_dict'].items():
        renamed_ckpt['state_dict'][k.replace('module.', '')] = v
    restore_net(net, renamed_ckpt)
    net = net.eval().cuda()
    return net