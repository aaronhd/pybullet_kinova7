"""Runs a random policy for the random object KinovaDiverseObjectEnv.
"""

import os, inspect
import numpy as np

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

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


def main():
    rospy.init_node('pybullet_kinova7')
    depth_pub = rospy.Publisher('pybullet/img/depth_raw', Image, queue_size=1)

    env = KinovaDiverseObjectEnv(renders=True, isDiscrete=False, maxSteps=2)
    policy = ContinuousDownwardBiasPolicy()
    # obs, done = env.reset(), False
    episode_rew = 0
    countStep = 0
    # obs, done = env.reset(), False
    # while True:
    #     pass
    while not rospy.is_shutdown():
          obs, done = env.reset(), False
          depth_image = bridge.cv2_to_imgmsg(obs)
          depth_pub.publish(depth_image)    
          # print("===================================")        
          # print("obs")
          # print(obs)
          countStep+=1
          while not done:
              env.render(mode='human')
              act = policy.sample_action(obs, 0.1)
              act[0] += 3 +0.46
              # print("Action")
              # print(act)
              obs, rew, done, _ = env.step(act)
              episode_rew += rew
          print("Episode reward", episode_rew, countStep)
    rospy.spin()


if __name__ == '__main__':
    main()
