# pybullet_kinova7

##Dependence

Anaconda2 python2

Tensorflow 1.7

ROS Kinetic

Keras 2.2.4

Pybullet 2.1.3

Cudnn 7.6.0

CUDA 9.0

Numpy 1.14.0

##Usage
1. Download the URDF file in the link.txt.

2. Download the UG-Net model, epoch08 for run_ugnet_pybullet, epoch10 for run_ugnet_greyd_pybullet.

3. roscore

4. rosrun yumi_grasp run_ugnet_pybullet.py  (rosrun rviz rviz FOR visualization)

5. python enjoy_kinova7_diverse_object_grasping.py （need to wait completion of loading model file in step4）
