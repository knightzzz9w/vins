import rospy
from std_msgs.msg import Header
import numpy as np
from collections import defaultdict
from sensor_msgs.msg import Image
from utils.convert_stamp import stamp2seconds
from gaussian_renderer import render
import torch
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
from utils.math_utils import inverse_Tmatrix
from utils.loss_utils import l1_loss, ssim
from scene import Scene, GaussianModel
import cv2
import time
from utils.camera_utils import loadCam
from sklearn.neighbors import NearestNeighbors
from munch import munchify
from gaussian_renderer import render
import open3d as o3d

class Estimator:
    
    def __init__(self , config):

        T_matrix = np.array(config["T_BS"]["data"]).reshape(4,4)
        self.ric = T_matrix[ :3 ,  :3 ]
        self.tic = T_matrix[ :3 ,  3 ]
        self.Tic = T_matrix
        
        self.deltatime = 1e-2
        
        self.width = config["resolution"][0]
        self.height = config["resolution"][1]
        
        self.K = np.array([[config["intrinsics"][0], 0, config["intrinsics"][2]],
                           [0, config["intrinsics"][1], config["intrinsics"][3]],
                           [0, 0, 1]])
        #param in gaussian
        
        self.dist_coeffs = np.array(config["distortion_coefficients"])
        
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        
        self.sh_degree = config["sh_degree"]
        self.gaussians = None
        self.frame_id = 0
        self.point_size = config["point_size"]
        
        self.pipeline_params = munchify(config["pipeline_params"])
        self.background = config["bg_color"]
        self.opt_params = munchify(config["opt_params"])
        self.render_pub = rospy.Publisher('gsrender', Image, queue_size=10)
        self.downsample_voxelsize = config["downsample_voxelsize"]
        self.downsample_factor = config["downsample_factor"]
    
        self.update_freq = config["update_freq"]
        self.update_offset = config["update_offset"]
        
        self.gaussian_th = config["Training"]["init_gaussian_th"]
        self.gaussian_extent = config["Training"]["init_gaussian_extent"]
        
        self.size_threshold = config["Training"]["size_threshold"]
        
        self.last_depth = 0.
            
        
    def dataalign(self , image_buf , keypose_buf , key_point_cloud_buf): #align the data k1 k2 p1 p2 k3
        
        measurements = []
        
        while(True):
            if(image_buf.empty() or keypose_buf.empty() or key_point_cloud_buf.empty()):
                return measurements
            
            if(stamp2seconds(image_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])  or  stamp2seconds(key_point_cloud_buf.queue[-1]) < stamp2seconds(keypose_buf.queue[0])):  #smaller than the first keypose
                return measurements
            
            if(stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(image_buf.queue[0])  and  stamp2seconds(keypose_buf.queue[0]) < stamp2seconds(key_point_cloud_buf.queue[0])):
                keypose_buf.get()
                continue
                
            while(stamp2seconds(keypose_buf.queue[0])   - stamp2seconds(key_point_cloud_buf.queue[0]) > self.deltatime):  #pop the data before it
                key_point_cloud_buf.get()
                
            while(stamp2seconds(keypose_buf.queue[0]) - stamp2seconds(image_buf.queue[0])   > self.deltatime): #pop the data
                image_buf.get()    #find the aligned data
            
            cur_image = image_buf.get()  ; cur_keypose = keypose_buf.get() ; cur_key_point_cloud = key_point_cloud_buf.get()
            
            time_cur_image = stamp2seconds(cur_image) ; time_cur_keypose = stamp2seconds(cur_keypose) ; time_cur_key_point_cloud = stamp2seconds(cur_key_point_cloud)
            #print("image time is " , time_cur_image , "keypose time is " , time_cur_keypose , "key_point_cloud time is " , time_cur_key_point_cloud)
            assert(abs(time_cur_image - time_cur_keypose) < self.deltatime) ; assert(abs(time_cur_image - time_cur_key_point_cloud) < self.deltatime)
            measurements.append([cur_image , cur_keypose , cur_key_point_cloud])
            
        return measurements
    
    
    def convert_ros_image_to_cv2(self, ros_image):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
            return cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return None
        
        
    def init_gaussian(self , training_args):
        self.gaussians = GaussianModel(self.sh_degree)
        training_args = munchify(training_args)
        self.gaussians.init_lr(training_args.spatial_lr_scale)
        self.gaussians.training_setup(training_args = training_args)
        
        
    def getpicdepth(self ,  points_2D , points_3D_cam ):
        
        # 初始化深度图
        depth_map = np.full((self.height, self.width), np.nan)

        # 提取深度值
        depths = points_3D_cam[:, 2]

        # 将已知深度值填入深度图
        for idx, (x, y) in enumerate(points_2D):
            depth_map[y,x] = depths[idx]

        # 构建KNN模型
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(points_2D)

        # 创建一个新的深度图以存储结果
        new_depth_map = np.copy(depth_map)

        # 遍历所有像素点
        for i in range(self.height):
            for j in range(self.width):
                if np.isnan(new_depth_map[i, j]):  # 如果点没有初始深度信息
                    # 获取半径为3的邻域
                    top = max(0, i-4) ; bot = min(self.height, i+5) ; left = max(0, j-4) ; right = min(self.width, j+5)
                    neighborhood = new_depth_map[top:bot, left:right]
                    if np.all(np.isnan(neighborhood)):  #如果这个点没有被填充
                        # 使用KNN计算该点的深度值
                        distances, indices = knn.kneighbors([[j, i]])
                        neighbor_depths = depths[indices[0]]
                        weights = 1 / (distances[0] + 1e-8)  # 防止除零
                        weighted_sum = np.sum(weights * neighbor_depths)
                        sum_weights = np.sum(weights)
                        new_depth_map[top:bot, left:right] = 0  # 标记为这个区域已填充 但是这些点是没有深度的 只有一个地方有深度
                        new_depth_map[i, j] = weighted_sum / sum_weights

        # 使用 cv2.inpaint 修复深度图中的空缺点
        mask = np.isnan(new_depth_map).astype(np.uint8)
        new_depth_map[np.isnan(new_depth_map)] = 0  # 将 NaN 替换为 0
        #new_depth_map_filled = cv2.inpaint(new_depth_map.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u = u.flatten()
        v = v.flatten()
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        
        depths = new_depth_map.flatten()
        valid_mask = depths > 0
        valid_pixel_coords = homogeneous_pixel_coords[:, valid_mask]
        valid_depths = depths[valid_mask]
        
        
        cam_coords = np.linalg.inv(self.K) @ (valid_pixel_coords * valid_depths)
        new_points_2D = valid_pixel_coords[:2].T
        new_points_3D_cam = cam_coords.T

        return new_points_2D , new_points_3D_cam

    def getnewpoints(self ,  points_2D , points_3D_cam ):

        depth_map = np.full((self.height, self.width), np.nan)

        depths = points_3D_cam[:, 2]
        depth_median = np.mean(depths)  #mean depth 
        self.last_depth = depth_median
        depth_std = 0.4
        
        # 将已知深度值填入深度图
        x_coords = points_2D[:, 0]
        y_coords = points_2D[:, 1]
        depth_map[y_coords, x_coords] = depths
        
        
        invalid_depth_mask = np.isnan(depth_map)
        depth_map[invalid_depth_mask] = depth_median
        depth_map += np.random.standard_normal(depth_map.shape)*np.where(invalid_depth_mask , depth_std*0.5 , depth_std*0.1 )
        u, v = np.meshgrid(np.arange(0 , self.width,7), np.arange(0 , self.height , 7))
        u = u.flatten() ; v = v.flatten()
        
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        filterdepth_map = depth_map[::7 , ::7]
        filterdepth_map = filterdepth_map.flatten()
        
        cam_coords = np.linalg.inv(self.K) @ (homogeneous_pixel_coords * filterdepth_map)
        new_points_3D_cam = cam_coords.T
        
        print("new points sample after downsampling is " , new_points_3D_cam.shape)
        return homogeneous_pixel_coords[:2 , :].T , new_points_3D_cam
    
    def getallnewpoints(self,last_median_depth):

        depth_map = np.full((self.height, self.width), last_median_depth)

        depth_std = 0.6
        depth_map[: , :] += np.random.standard_normal(depth_map.shape)*depth_std*0.5
        u, v = np.meshgrid(np.arange(0 , self.width,7), np.arange(0 , self.height , 7))
        u = u.flatten() ; v = v.flatten()
        
        homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
        filterdepth_map = depth_map[::7 , ::7]
        filterdepth_map = filterdepth_map.flatten()
        
        cam_coords = np.linalg.inv(self.K) @ (homogeneous_pixel_coords * filterdepth_map)
        new_points_3D_cam = cam_coords.T
        
        print("new points sample after downsampling is " , new_points_3D_cam.shape)
        return homogeneous_pixel_coords[:2 , :].T , new_points_3D_cam



    def pub_render(self , tensor_image):
        
        array = tensor_image.cpu().detach().numpy()
        # 由于sensor_msgs/Image期望图像数据为uint8，因此需要转换类型
        array = (array * 255).astype(np.uint8)

        # 如果tensor是3D的，将其转换为4D (H, W, C)
        if len(array.shape) == 3:
            array = array.transpose(1, 2, 0)
        array = array[:, :, ::-1]
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(array, encoding="passthrough")
        self.render_pub.publish(ros_image)
        

    
    def project_and_get_colors(self, measurements):
        for cur_image, cur_keypose, cur_key_point_cloud in measurements:
            # 假设 cur_key_point_cloud 是 Nx3 的点云数组
            points_3D_ext = np.array([[point.x, point.y, point.z , 1] for point in cur_key_point_cloud.points])  # Nx4  扩展的
            print("cur point size is %d"  , points_3D_ext.shape[0])
            
            t_pre1 = time.time()
            # 从cur_keypose获取旋转和平移矩阵
            position = cur_keypose.pose.pose.position  #R_wbk
            orientation = cur_keypose.pose.pose.orientation   #T_wbk
            
            

            translation = np.array([position.x, position.y, position.z])
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            
            T_Wbk = np.eye(4)
            T_Wbk[ :3 ,  :3 ] = rotation_matrix ; T_Wbk[ :3 , 3 ] = translation
            
            T_Wck = T_Wbk@self.Tic
            
            T_ckW = inverse_Tmatrix(T_Wck)
            assert( np.sum(np.abs((T_ckW@T_Wck - np.eye(4)))) < 1e-3  )
            if points_3D_ext.shape[0] > 0:                
                
                t_pre_part1_1 = time.time()
                
                points_3D_ext_cam = points_3D_ext@T_ckW.T
                points_3D_cam = points_3D_ext_cam[: , :3]  #得到在当前相机坐标系下表示的点云
                
                
                points_2D_cam = points_3D_cam@self.K.T #还要必上Z
                points_2D = points_2D_cam[:, :2] / (points_2D_cam[:, 2].reshape(-1, 1))  # Nx2
                print("num of point is " , points_2D.shape[0])
                
                t_pre_part1_2 = time.time()
                print("perpare 2D points time  is " , (t_pre_part1_2 - t_pre_part1_1)*1e3, " ms")
                
                #print(points_2D)
                
                t_pre_part2_1 = time.time()
                points_2D = np.round(points_2D).astype(int)
                filtered_ids = (
                    (points_2D[:, 0] >= 0) & (points_2D[:, 0] < self.width) &
                    (points_2D[:, 1] >= 0) & (points_2D[:, 1] < self.height)
                )
                points_2D = points_2D[filtered_ids] ; points_3D_cam = points_3D_cam[filtered_ids] ; points_3D_ext = points_3D_ext[filtered_ids]
                print("num of point in screen is " , points_2D.shape[0])
                #self.getpicdepth(points_2D , points_3D_cam )  时间太慢了 去掉
                t_pre_part2_2 = time.time()
                print("perpare legal 2D points time  is " , (t_pre_part2_2 - t_pre_part2_1)*1e3, " ms")
                
                t_addpoints = time.time()
                new_point2D , new_point3D_cam = self.getnewpoints(points_2D , points_3D_cam)  #new point in cam cor
                t_addpoints2 = time.time()
                print("add points time is " , (t_addpoints2 - t_addpoints)*1e3 , "ms")
            
            
            else:
                new_point2D , new_point3D_cam = self.getallnewpoints(self.last_depth)
            
            
            t_pre_part3_1 = time.time()
            image = self.convert_ros_image_to_cv2(cur_image)  #图像去掉畸变
            #print(image.shape)
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  #shape is [height , width , 3]  color is bgr
            # colors = []
            # for point in points_2D:
            #     x, y = point[0], point[1] #分别代表横向的和纵向的
            #     cur_color = image[y, x]
            #     rgb_colors = [float(cur_color[2]/255) , float(cur_color[1]/255), float(cur_color[0]/255)]
            #     colors.append(rgb_colors)
            
            x_coords = new_point2D[:, 0]
            y_coords = new_point2D[:, 1]
            selected_colors = image[y_coords, x_coords]
            normalized_colors = selected_colors.astype(float) / 255.0

            # 将颜色顺序从 BGR 转换为 RGB
            normalized_colors = normalized_colors[:, ::-1].copy()
            print(normalized_colors.shape)
            
            t_pre_part3_2 = time.time()
            print("perpare 2D points color time  is " , (t_pre_part3_2 - t_pre_part3_1)*1e3, " ms")
            
            
            t_pre2 = time.time()
            print("perpare time is " , (t_pre2 - t_pre1)*1e3, " ms")
            
            
            
            
            
            cam_model = loadCam(self.frame_id , image , T_Wck[:3 , :3] ,T_Wck[:3 , 3] , self.K  )
            
            t_add_gaussian = time.time()
            R_Wck = T_Wck[:3 , :3] ; t_Wck = T_Wck[:3 , 3]
            points_3D = new_point3D_cam@R_Wck.T + t_Wck
            self.gaussians.extend_from_pcd_seq(cam_info = cam_model , point_3D = points_3D  , colors = normalized_colors , point_size = self.point_size , kf_id = self.frame_id)
            t_add_gaussian2 = time.time()
            print("add gaussian time is " , (t_add_gaussian2 - t_add_gaussian)*1e3 , "ms" )
            
            
            
            
            t_render = time.time()
            render_pkg = render(
                cam_model, self.gaussians, self.pipeline_params, torch.tensor(self.background).to("cuda")
            )
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]            
            t_render2 = time.time()
            
            print("rendering time is " , (t_render2 - t_render)*1e3 , "ms" )
            
            # Loss
            
            
            t_backward = time.time()
            gt_image = cam_model.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            #iter_end.record()
            
            t_backward2 = time.time()
            print("backward time is " , (t_backward2 - t_backward)*1e3 , "ms" )
            
            print("cur gaussian num is " , (self.gaussians._xyz.shape))
            
            self.pub_render(image)
            
            if self.frame_id % self.update_freq == self.update_offset:
                self.gaussians.densify_and_prune(
                self.opt_params.densify_grad_threshold,
                self.gaussian_th,
                self.gaussian_extent,
                self.size_threshold,
                    )
                
            self.frame_id += 1
            
            
            
            

            
    
    
    def get_measurements(self , image_buf , keypose_buf , key_point_cloud_buf):
        return self.dataalign( image_buf , keypose_buf , key_point_cloud_buf)
    
    
    
    
            
        


