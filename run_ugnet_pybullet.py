#! /usr/bin/env python

import os
import time
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

import cv2
import scipy.ndimage as ndimage
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage.feature import peak_local_max

import rospy
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import math


# some message type
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

os.environ["CUDA_VISIBLE_DEVICES"]="0"

bridge = CvBridge()

# Load the Network.
print("Tensorflow Version: %s" % str(tf.__version__))
# comment by Aaron
# MODEL_FILE = '/home/aarons/catkin_kinect/src/yumi_grasp/src/epoch_10_model.hdf5'  # UG-Net
# MODEL_FILE = '/home/aarons/catkin_kinect/src/yumi_grasp/src/epoch_97_model.hdf5' #ggcnn
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181210_2003__u-net_9_5_3__32_16_8/epoch_10_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181210_1437__u-net_9_5_3__32_16_8/epoch_10_model.hdf5'  # muqian zuihao
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181215_1409__u-net_9_5_3__32_16_8/epoch_10_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181216_1527__u-net_9_5_3__32_16_8/epoch_10_model.hdf5'  #zhengcheng le
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181219_2252__UG-Net/epoch_15_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181220_1737__UG-Net/epoch_10_model.hdf5'  # hao

# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181223_2227__UG-Net/epoch_10_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181225_1902__UG-Net/epoch_05_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181227_2156__UG-Net/epoch_05_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181228_1136__UG-Net/epoch_12_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181228_2042__UG-Net/epoch_12_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181229_0024__UG-Net/epoch_08_model.hdf5'
MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181229_1049__UG-Net/epoch_08_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181229_1237__UG-Net/epoch_10_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181229_1542__UG-Net_lightv2/epoch_08_model.hdf5'

# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181229_2044__UG-Net/epoch_06_model.hdf5' #0104
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181230_1053__UG-Net_lightv2/epoch_08_model.hdf5'
# MODEL_FILE = '/media/aarons/hdd_2/ggcnn_model/networks/181230_2032__UG-Net_lightv2/epoch_06_model.hdf5'


model = load_model(MODEL_FILE)

print(MODEL_FILE)


rospy.init_node('ggcnn_detection')

# Output publishers.
grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
ang_pub_heatmap = rospy.Publisher('ggcnn/img/angheatmap', Image, queue_size=1)
width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)

grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
depth_raw_pub = rospy.Publisher('ggcnn/img/depth_raw', Image, queue_size=1)

ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
pointout_pub = rospy.Publisher('ggcnn/out/points_out', Image, queue_size=1)

# pointout_pub = rospy.Publisher('ggcnn/out/points_out', numpy_msg(Floats), queue_size=1)



# Initialise some globals.
prev_mp = np.array([150, 150])
ROBOT_Z = 0
Input_Res = 304
crop_size = 304 #400  330
VISUALISE = True


# Tensorflow graph to allow use in callback.
graph = tf.get_default_graph()

# Get the camera parameters
# camera_info_msg = rospy.wait_for_message('/kinect2/qhd/camera_info', CameraInfo)  #0321
# camera_info_msg = rospy.wait_for_message('/kinect2/sd/camera_info', CameraInfo)


# K = camera_info_msg.K
fx = 1
cx = 1
fy = 1
cy = 1

# print(fx)
# print(cx)
# print(fy)
# print(cy)

# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = True

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s time: %s' % (self.s, self.t1 - self.t0))

def visu_heatmap(data):
    global Input_Res
    out_img = np.zeros((Input_Res, Input_Res, 3), dtype=np.uint8)
    fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='jet', xticklabels=False, yticklabels=False, cbar=False)
    fig.add_axes(ax)
    fig.canvas.draw()
    data_heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data_heatmap = data_heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data_crop = cv2.resize(data_heatmap[(480-crop_size)//2:(480-crop_size)//2+crop_size, \
        (640-crop_size)//2:(640-crop_size)//2+crop_size,:], (Input_Res, Input_Res))
    out_img[:, :, 0] = data_crop[:,:,2]    # R
    out_img[:, :, 1] = data_crop[:,:,1]    # G
    out_img[:, :, 2] = data_crop[:,:,0]    # B
    return out_img




def robot_pos_callback(data):
    global ROBOT_Z
    ROBOT_Z = data.pose.position.z


def depth_callback(depth_message):
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy
    global crop_size
    with TimeIt('prediction'):
    # with TimeIt('Crop'):
        depthImg = bridge.imgmsg_to_cv2(depth_message)
        rospy.loginfo(depthImg.shape)


        near = 0.01
        far = 0.24
        depth = far * near / (far - (far - near) * depthImg)


        # Crop a square out of the middle of the depth and resize it to 300*300
        depth_crop = cv2.resize(depth[(304-crop_size)//2:(304-crop_size)//2+crop_size, (304-crop_size)//2:(304-crop_size)//2+crop_size], (304, 304))
        # print((424-crop_size)//2, (424-crop_size)//2+crop_size)
        # print((512-crop_size)//2, (512-crop_size)//2+crop_size)
        # exit()

        depth_crop = cv2.resize(depth, (Input_Res, Input_Res))


        # Replace nan with 0 for inpainting.
        depth_crop = depth_crop.copy()
        depth_nan = np.isnan(depth_crop).copy()
        # print(depth_nan)
        depth_crop[depth_nan] = 0
        # np.save("/home/aarons/catkin_kinect/src/yumi_grasp/src/depth_raw_pub2.npy", depth_crop)




    # with TimeIt('Inpaint'):
        # open cv inpainting does weird things at the border.
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        mask = (depth_crop == 0).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32)/(depth_scale)  # Has to be float32, 64 not supported.
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale / 1.0  # kinect output unit is millemeter, but realsense output unit is meter

        # kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        # depth_crop=cv2.morphologyEx((depth_crop * 1000),cv2.MORPH_OPEN,kernel)
        # depth_crop = depth_crop /1000.0
        # depth_crop = cv2.bilateralFilter(depth_crop,7,31,31)




        # print(depth_crop.shape)
        # exit()



    # with TimeIt('Calculate Depth'):
        # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
        depth_crop_neighbor = depth_crop.copy()
        # depth_center = depth_crop[100:141, 130:171].flatten()
        depth_center = depth_crop.flatten()
        depth_center.sort()
        # depth_center = depth_center[:10].mean() * 1000.0
        depth_center = depth_center.mean() * 1000.0
        # print('depth_center')
        # print(depth_center)
        # if depth_center > 785:
        #     return

    # with TimeIt('Inference'):
        # Run it through the network.
        # np.save("/home/aarons/catkin_kinect/src/yumi_grasp/src/depth_raw_pub1.npy", depth_crop)
        depth_raw_pub.publish(bridge.cv2_to_imgmsg(depth_crop))
        # print(depth_crop.max())
        # print(depth_crop.mean())
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1.1) #1224
        # depth_crop = np.clip((depth_crop - depth_crop.max()), -1, 1.1) #1224
        # depth_crop = np.clip(np.true_divide((depth_crop - depth_crop.mean()), depth_crop.std()), -1, 1.1)
        # depth_crop = np.clip((depth_crop - aaa), -1, 1.1)

        # print(depth_crop.shape)
        # depth_crop = np.clip((depth_crop), -1, 1)


        with TimeIt('Inference'):
            with graph.as_default():
                # print("begin prediction")
                pred_out = model.predict(depth_crop.reshape((1, Input_Res, Input_Res, 1)))
                # print("end prediction")

        # print(np.shape(pred_out))
        points_out = pred_out[0].squeeze()
        # print(np.shape(points_out))
        # print(depth_nan.shape)
        # print(depth_nan)
        # Replace nan with 0 for inpainting.
        points_out[depth_nan] = 0





    # with TimeIt('Trig'):
        # Calculate the angle map.
        cos_out = pred_out[1].squeeze()
        sin_out = pred_out[2].squeeze()
        ang_out = np.arctan2(sin_out, cos_out)/2.0

        width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

    # with TimeIt('Filter'):
        # Filter the outputs.
        points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0   5.0 aaron
        ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)

    # with TimeIt('Control'):
        # Calculate the best pose from the camera intrinsics.
        maxes = None

        ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.

        if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            prev_mp = max_pixel.astype(np.int)
        else:
            # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
            # maxes = peak_local_max(points_out, min_distance=20, threshold_abs=0.1, num_peaks=20)  #min_distance=10, threshold_abs=0.1, num_peaks=3  15 0.1 20            
            maxes = peak_local_max(points_out, min_distance=5, threshold_abs=0.1, num_peaks=1)  #min_distance=10, threshold_abs=0.1, num_peaks=3  15 0.1 20
            if maxes.shape[0]:
                max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]
                # max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
                visual_max_pixel = max_pixel.copy()

            # Keep a global copy for next iteration.
                # prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)
                grasp_quality = points_out[max_pixel[0],max_pixel[1]] 
            else:
                rospy.loginfo("no lacal maxes! ")
                # grasp_quality = 0
                # cmd_msg = Float32MultiArray()
                # x = 0
                # y = 0
                # z = 0
                # ang = 0
                # width = 0
                # depth_grasp_neighbor = 0
                # grasp_quality = 0
                # cmd_msg.data = [x, y, z, ang, width, depth_grasp_neighbor, grasp_quality]
                # cmd_pub.publish(cmd_msg)
                # rospy.loginfo(cmd_msg)
                return


            # max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]
            # visual_max_pixel = max_pixel.copy()

            # # Keep a global copy for next iteration.
            # prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)
            # # print(max_pixel)
            # grasp_quality = points_out[max_pixel[0],max_pixel[1]]
        if max_pixel[0]>=10 and max_pixel[0]<=394 and max_pixel[1]>=10 and max_pixel[1]<=394:
            # print('bound exists! ')
            depth_grasp_neighbor = depth_crop_neighbor[max_pixel[0]-10:max_pixel[0]+10, max_pixel[1]-10:max_pixel[1]+10].flatten()
            depth_grasp_neighbor.sort()
            depth_grasp_neighbor = depth_grasp_neighbor[:50].mean() * 1000.0
            # print(depth_grasp_neighbor)
        else:
            depth_grasp_neighbor = depth_center


        ang = ang_out[max_pixel[0], max_pixel[1]]
        width = width_out[max_pixel[0], max_pixel[1]]
        if (depth_center - depth_grasp_neighbor) < 5:
            rospy.loginfo('task space is empty!')
            print(depth_center - depth_grasp_neighbor)
            grasp_quality = 0
        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        # max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(424 - crop_size)//2, (512 - crop_size) // 2]))
        max_pixel = ((np.array(max_pixel) / 304.0 * crop_size) + np.array([(304 - crop_size)//2, (304 - crop_size) // 2]))  #[2,1]
        # depth_crop = cv2.resize(depth[(540-crop_size)//2:(540-crop_size)//2+crop_size, (960-crop_size)//2:(960-crop_size)//2+crop_size], (300, 300))
        # where is orign point

        max_pixel = np.round(max_pixel).astype(np.int)

        point_depth = depthImg[max_pixel[0], max_pixel[1]]

        # print('point_depth')
        # print(point_depth)

        view_matrix = np.array([[0.0, 1.0, -0.0, 0.0],[-1.0, 0.0, -0.0, 0.0],[0.0, 0.0, 1.0, 0.0], [-0.0, -0.6499999761581421, -1.2400000095367432, 1.0]])  
        proj_matrix = np.array([[4.510708808898926, 0.0, 0.0, 0.0], [0.0, 4.510708808898926, 0.0, 0.0],[ 0.0, 0.0, -1.0020020008087158, -1.0], [0.0, 0.0, -0.0200200192630291, 0.0] ])
        inter_gl = np.dot(view_matrix, proj_matrix)
        # inter_gl = np.dot(proj_matrix, view_matrix)

        px = 2.0*(max_pixel[1] - 0)/304.0 - 1.0
        py = 1.0 - (2.0*max_pixel[0])/304.0
        # py = 2.0*(max_pixel[0] - 0)/304.0 - 1.0
        pz = 2.0*point_depth - 1.0 
        PP3D = np.array([px, py, pz, 1.0])

        PP_world = np.dot(PP3D, np.linalg.inv(inter_gl))
        # PP_world = np.dot( np.linalg.inv(inter_gl), PP3D)
        rospy.loginfo("PP_world")
        print(PP3D)
        print(PP_world)
        print(PP_world/PP_world[3])
        print(depth[max_pixel[0], max_pixel[1]])

        # These magic numbers are my camera intrinsic parameters.
        # fov = 25.0
        # fx = 1/math.tan(math.radians(fov/2.0))
        # theta_x = math.radians(-90) + math.atan2(max_pixel[1],fx)
        # X = point_depth * math.tan(theta_x)
        # Z = point_depth
        # fy = 1/math.tan(math.radians(fov/2.0))
        # theta_y = math.atan2(max_pixel[0],fy)
        # Y = math.tan(theta_y) * point_depth/ math.cos(theta_x)
        # print([X, Y, Z])



        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth
        if np.isnan(z):
            print("depth is nan!")
            return

        # print("predict: ")
        # print(x, y, z, ang, width, depth_center)

    # with TimeIt('Draw'):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        grasp_img = np.zeros((Input_Res, Input_Res, 3), dtype=np.uint8)
        # with open('/home/aarons/catkin_kinect/src/yumi_grasp/src/heatmap.pkl', 'w') as f:
        #     pickle.dump(points_out, f)
        # print(points_out.shape)
        # np.save("/home/aarons/catkin_kinect/src/yumi_grasp/src/points_out_1229_householdclutter.npy", points_out)
        # np.savetxt("/home/aarons/catkin_kinect/src/yumi_grasp/src/light_txt.npy", points_out)
        # exit()
        # pd_pointout = pd.DataFrame(points_out)
        # pd_pointout.to_csv('/home/aarons/catkin_kinect/src/yumi_grasp/src/points_out.csv')

        # heatmap test code
        # fig, ax = plt.subplots()
        # ax = sns.heatmap(ang_out, cmap='jet', xticklabels=False, yticklabels=False, cbar=False)
        # fig.add_axes(ax)
        # fig.canvas.draw()
        # data_heatmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data_heatmap = data_heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # data_crop = cv2.resize(data_heatmap[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size,:], (Input_Res, Input_Res))
        # print(data_crop.shape)
        if VISUALISE :
            # pointout_visu = points_out.flatten().tolist()
            # print(len(pointout_visu))
            # bridge.cv2_to_imgmsg(points_out)

            pointout_pub.publish(bridge.cv2_to_imgmsg(points_out))  # for visualization module


            # pose_heatmap = visu_heatmap(points_out)
            # grasp_img = pose_heatmap


            # ang_heatmap = visu_heatmap(ang_out)
            # ang_img = ang_heatmap
            # width_heatmap = visu_heatmap(width_out)
            # width_img = width_heatmap

            # grasp_img[:,:,2] = (points_out * 255.0)
            grasp_img_plain = grasp_img.copy()
            grasp_img[:,:,2] = (points_out * 255.0)
            # rr, cc = circle(prev_mp[0], prev_mp[1], 5)
            rr, cc = circle(visual_max_pixel[0], visual_max_pixel[1], 5)

            # depth_crop[rr, cc] = 200
            grasp_img[rr, cc, 0] = 0    # R
            grasp_img[rr, cc, 1] = 255  # G
            grasp_img[rr, cc, 2] = 0    # B
        # with TimeIt('Publish'):
            # Publish the output images (not used for control, only visualisation)
            # grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')

            grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
            grasp_img.header = depth_message.header
            grasp_pub.publish(grasp_img)
            


            # ang_img = bridge.cv2_to_imgmsg(ang_img, 'bgr8')
            # ang_img.header = depth_message.header
            # ang_pub_heatmap.publish(ang_img)

            # width_heatmap = bridge.cv2_to_imgmsg(width_heatmap, 'bgr8')
            # width_heatmap.header = depth_message.header
            # width_pub.publish(width_heatmap)

            grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
            grasp_img_plain.header = depth_message.header
            grasp_plain_pub.publish(grasp_img_plain)

            depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))
            ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x, y, z, ang, width, depth_grasp_neighbor, grasp_quality, depth_center, visual_max_pixel[0], visual_max_pixel[1] ]
        rospy.loginfo(cmd_msg)
        cmd_pub.publish(cmd_msg)


depth_sub = rospy.Subscriber('pybullet/img/depth_raw', Image, depth_callback, queue_size=1)

## comment by Aaron
# robot_pos_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', PoseStamped, robot_pos_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()
