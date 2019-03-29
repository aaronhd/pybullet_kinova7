"""Runs a random policy for the random object KinovaDiverseObjectEnv.
"""

import os, inspect
import numpy as np
import gym
import gym.spaces
import cv2
from cv_bridge import CvBridge
from kinova_diverse_object_gym_env import KinovaDiverseObjectEnv
from gym import spaces
import rospy
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
# some message type
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray


# global value
command_list =[]
state_list=[False]
new_command = False

bridge = CvBridge()

class ContinuousDownwardBiasPolicy(object):
  """Policy which takes continuous actions, and is biased to move down.
  """
  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.

    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-3, high=3, shape=(5,))

  def sample_action(self, obs, explore_prob):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da, close = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da, 0]

def control_callback(control_list):
    global command_list
    global new_command
    if control_list.data[0] == 0 and control_list.data[1] == 0 and control_list.data[2] == 0:
        # rospy.loginfo("invalid control point")
        pass
        # rospy.loginfo(control_list)
    else:
      if new_command is False:
        command_list = control_list.data
        new_command = True

def state_callback(state):
    global state_list
    state_list = state.data
    # print("state_list")
    # print(state_list[0])

def main():
    global command_list
    global new_command
    global state_list
    rospy.init_node('pybullet_kinova7')
    depth_pub = rospy.Publisher('pybullet/img/depth_raw', Image, queue_size=1)
    control = rospy.Subscriber('ggcnn/out/command', Float32MultiArray, control_callback, queue_size=1)
    percept_state = rospy.Subscriber('ggcnn/out/state', Float32MultiArray, state_callback, queue_size=1)

    env = KinovaDiverseObjectEnv(renders=True, isDiscrete=False, maxSteps=3, rgbd_used=False)
    # policy = ContinuousDownwardBiasPolicy()
    episode_rew = 0
    countStep = 0
    while not rospy.is_shutdown():
        obs, done = env.reset(), False
        rospy.loginfo("New Try !")
        print(obs.shape)
        depth_image = bridge.cv2_to_imgmsg(obs)
        depth_pub.publish(depth_image)    
        countStep+=1
        while not done:
            env.render(mode='human')
            # #  for exploration 
            # act = policy.sample_action(obs, 0.1)
            # act[0] += 3 +0.46
            # print("Action")
            # # print(act)
            # # action [x, y, z, angle, width]
            if new_command:
              # print("command list")
              # print(command_list)
              # rospy.loginfo(len(command_list))
              act = command_list
              obs, rew, done, _ = env.step(act)
              episode_rew += rew
              if done:
                new_command = False
              print("done", done)
            if state_list[0]==1:
              countStep-=1
              break
        print("Episode reward", episode_rew, countStep)
    rospy.spin()

if __name__ == '__main__':
    main()
