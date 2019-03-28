from kinovaGymEnv import KinovaGymEnv
import random
import os
from gym import spaces
import time
import pybullet as p
import kinova
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym
import copy
import math

import rospy
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

# some message type
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray


class KinovaDiverseObjectEnv(KinovaGymEnv):
  """Class for Kinova environment with diverse objects.

  In each episode some objects are chosen from a set of 1000 diverse objects.hi
  These 1000 objects are split 90/10 into a train and test set.
  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1500,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=4,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.1,  #0.3
               cameraRandom=0,
               width=304,
               height=304,
               numObjects=1,
               isTest=False):
    """Initializes the KinovaDiverseObjectEnv. 

    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180 
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._isTest = isTest
    self._obj_pos =[]
    self.rgbd_used = True

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid<0):
         self.cid = p.connect(p.GUI)
      # p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
      p.resetDebugVisualizerCamera(1.3,47.6,-22.6,[0.47,-0.0,0.19])

    else:
      self.cid = p.connect(p.DIRECT)
    self._seed()
    # self._reset()

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,))  # dx, dy, dz, da
    self.viewer = None

  def _reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    look = [0.65, 0.0, 0.24]  #0.23 0.2
    distance = 1
    pitch = -90 + self._cameraRandom*np.random.uniform(-3, 3)
    yaw = -90 + self._cameraRandom*np.random.uniform(-3, 3)
    roll = 0
    # print("rotation list:")
    # print(pitch , yaw, roll)
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
        look, distance, yaw, pitch, roll, 2)
    fov = 25. + self._cameraRandom*np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(
        fov, aspect, near, far)
    
    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)

    # 3.16
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    # p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-0.6320000,0.000000,0.000000,0.0,1.0)

    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-0.630000,0.000000,0.000000,0.0,1.0)
    # p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
            
    p.setGravity(0,0,-10)
    self._kinova = kinova.Kinova(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    urdfList = self._get_random_object(
      self._numObjects, self._isTest)
    self._objectUids = self._randomly_place_objects(urdfList)
    self._observation = self._get_observation()
    return np.array(self._observation)

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.

    Args:
      urdfList: The list of urdf files to place in the bin.

    Returns:
      The list of object unique ID's.
    """
    # Randomize positions of each object urdf.
    objectUids = []
    for urdf_name in urdfList:
      xpos = 0.6 +self._blockRandom*random.random()  #0.55
      ypos = 0 +self._blockRandom*(random.random()-0.5)  #0.5
      angle = np.pi/2 + self._blockRandom * np.pi * random.random()
      orn = p.getQuaternionFromEuler([0, 0, angle])
      urdf_path = os.path.join(self._urdfRoot, urdf_name)
      # urdf_path = "/home/aarons/sim2real7/pybullet_data/bar_clamp/urdf/bar_clamp.urdf"
      print("urdf_path")
      print(urdf_path)
      uid = p.loadURDF(urdf_path, [xpos, ypos, 0.08],
        [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)

      # print(p.getBasePositionAndOrientation(uid)[0])
      # print("physical pose")
      # self._obj_pos = [xpos, ypos, .08 , 0.0, 0]
      # print(self._obj_pos)
      # print(":_randomly_place_objects")
      # print([xpos, ypos, .15])
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(500):
        p.stepSimulation()
    return objectUids

  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = p.getCameraImage(width=self._width,
                                      height=self._height,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    depthImg = np.expand_dims(img_arr[3], -1)

    if self.rgbd_used:
      rgbd = np.concatenate((rgb[:,:,:3], depthImg), axis=2)
      return rgbd
    else:
    # near = 0.01
    # far = 0.24
    # depth = far * near / (far - (far - near) * depthImg)
    # np_depth_arr = np.reshape(depth, (self._height, self._width))
    # return np_img_arr[:, :, :3]
      return depthImg

  def _step(self, action):
    """Environment step.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # print("step function")
    # print(action)
    return self._step_continuous([action[0], action[1], action[2], action[3], action[4]])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    # print("_step_continuous")
    # print(action)

    # get current robot state
    state = p.getLinkState(self._kinova.kinovaUid,
                                  self._kinova.kinovaEndEffectorIndex)
    end_effector_pos = state[0]

    # test_ang = 1.57
    action2 = [action[0], action[1], action[2] + 0.35, action[3], 0]
    # print("end_effector_pos[2]")
    # print(end_effector_pos[2], action2[2])
    self._env_step += 1
    print("_env_step: %d of %d" %(self._env_step,self._maxSteps))
    # print(self._env_step, self._maxSteps)
    # print(action2)

    # If we are close to the bin, attempt grasp.
    if end_effector_pos[2] >= action2[2]+0.05:
      self._kinova.applyAction(action2, 1)
      print("Approaching!")
      for _ in range(self._actionRepeat):
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        if self._termination():
          break
   
    else:
    # if end_effector_pos[2] <= action2[2]+0.10:
      print("Start grasping!")
      finger_angle = 0.5
      z_value = end_effector_pos[2]
      for _ in range(500):
        # print("Downing! !")
        # print(z_value)
        grasp_action = [action2[0], action2[1], z_value, action2[3], finger_angle]
        # print(grasp_action)
        self._kinova.applyAction(grasp_action, 1)
        p.stepSimulation()
        z_value -= 0.35 / 100
        z_value = max(z_value, action[2], 0.04)   # depth for 0.06  rgbd for 0.04 
        if self._renders:
          time.sleep(self._timeStep)
      observation = self._get_observation()

      for _ in range(300):
        # print("Grasping ! ")
        grasp_action = [action2[0], action2[1], z_value , action2[3], finger_angle]
        # print(grasp_action)
        self._kinova.applyAction(grasp_action, 0)
        p.stepSimulation()
        finger_angle += 0.8/200.
        if finger_angle > 1.3:
          finger_angle = 1.3
        if self._renders:
          time.sleep(self._timeStep)
      observation = self._get_observation()

      for _ in range(500):
        # print("Lifting !")
        grasp_action = [action2[0], action2[1], z_value+0.3, action2[3], finger_angle]
        # print(grasp_action)
        self._kinova.applyAction(grasp_action, 1)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        self._attempted_grasp = True

    observation = self._get_observation()
    done = self._termination()
    reward = self._reward()

    debug = {
        'grasp_success': self._graspSuccess
    }
    return observation, reward, done, debug

  def _reward(self):
    """Calculates the reward for the episode.

    The reward is 1 if one of the objects is above height .2 at the end of the
    episode.
    """
    reward = 0
    self._graspSuccess = 0
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(
        uid)
      # print("reward:")
      # print(pos)
      # If any block is above height, provide reward.
      if pos[2] > 0.2:
        self._graspSuccess += 1
        reward = 1
        break
    return reward

  def _termination(self):
    """Terminates the episode if we have tried to grasp or if we are above
    maxSteps steps.
    """
    return self._attempted_grasp or self._env_step >= self._maxSteps

  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    # print("_get_random_object")
    # print(found_object_directories)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects),
                                        num_objects)
    # print(selected_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames
  
  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    
    reset = _reset
    
    step = _step