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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import cv2
import torch

WARNED = False

def loadCam( frame_id , bgr_image, R, T , K ):
    

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  #shape is (H , W , 3) color is rgb

    rgb_image = rgb_image.transpose(2, 0, 1)  #shape is (3 , H , W) color is rgb
    rgb_tensor = torch.from_numpy(rgb_image).float()  
    rgb_tensor = rgb_tensor / 255.0
    
    #get the fov
    fx = K[0 , 0] ; fy = K[1 ,1] ; cx = K[0,2] ; cy = K[1,2]

    return Camera(frame_id=frame_id, R=R, T=T, 
                  fx = fx , fy = fy , cx = cx , cy = cy , 
                  image=rgb_tensor)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
