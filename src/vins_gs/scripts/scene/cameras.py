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
import math
from utils.graphics_utils import getWorld2View, getProjectionMatrix2

class Camera(nn.Module):
    def __init__(self, frame_id, R, T, fx , fy,cx , cy   , image, 
                gt_alpha_mask=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.frame_id = frame_id
        self.R = R
        self.T = T
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View(R, T)).cuda()  #没有trans
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, cx = cx , cy = cy , fx = fx , fy = fy , W = self.image_width , H =self.image_height).cuda()
        self.full_proj_transform = self.projection_matrix@self.world_view_transform
        
        self.world_view_transform_T = self.world_view_transform.T
        self.projection_matrix_T = self.projection_matrix.T
        self.full_proj_transform_T = self.full_proj_transform.T
        
        
        
        self.camera_center = self.world_view_transform[:3, 3]
        
        self.FoVx = 2*math.atan(self.image_width/(2*fx))
        self.FoVy = 2*math.atan(self.image_height/(2*fy))

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

