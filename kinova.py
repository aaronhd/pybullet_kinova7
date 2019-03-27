import os,  inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data



class Kinova:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 2000.
    self.fingerAForce = 2 
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace =21
    self.useOrientation = 1
    self.kinovaEndEffectorIndex = 9
    self.kinovaGripperIndex = 7
    
    # lower limits for null space
    self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
    # upper limits for null space
    self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4,6,5.8,4,5.8]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0,0,0,0]
    #joint damping coefficents
    # self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
    self.jd=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    self._kinovaStartPos = [0,0,0]
    self.reset()
    
  def reset(self):
    kinova = p.loadURDF(os.path.join(self.urdfRootPath,"kinova_description2/urdf/j2n7s300_standalone.urdf"),self._kinovaStartPos,useFixedBase=1)
    #for i in range (p.getNumJoints(self.kinovaUid)):
    #  print(p.getJointInfo(self.kinovaUid,i))
    self.kinovaUid = kinova
    # p.resetBasePositionAndOrientation(self.kinovaUid,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])


    # for i in range(self.kinovaEndEffectorIndex+1):
    #   JointInfo = p.getJointInfo(self.kinovaUid,i)
    #   # print(JointInfo)
    #   #lower limits for null space
    #   self.ll[i] = JointInfo[8]
    #   #upper limits for null space
    #   self.ul[i] = JointInfo[9]
      #joint damping coefficents
      # jd[i] = JointInfo[6]
    # print(jd)





    # self.jointPositions=[ 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200, -0.000200]
    # self.jointPositions=[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # self.jointPositions=[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # self.jointPositions=[ 0, 0, -3.2, -1.0, 0.78, 3.31, 7.28, 2.2, 2.28, 0, -4.58, 0, 7.05, 0, -7.6, 0]
    # self.jointPositions=[0.0, 0.0, 1.691884506295414, 0.523596664758403, -3.3795658428691167, 0.7290620967898366, 1.6257430100542183,
    #                      -3.124612041350533, 0.0036354312522859397, 0.0, 1.7657059909627546e-05, 0.0, -6.28493691840157e-06, 0.0, -1.0767534552816044e-08, 0.0]
    self.numJoints = p.getNumJoints(self.kinovaUid)

    # for jointIndex in range (self.numJoints):
    # #   # p.resetJointState(self.kinovaUid,jointIndex,self.jointPositions[jointIndex])
    #   p.setJointMotorControl2(self.kinovaUid,jointIndex,p.POSITION_CONTROL,targetPosition=self.jointPositions[jointIndex],force=self.maxForce)


    #   print(p.getJointState(self.kinovaUid,jointIndex))    
    # print("======reset=========")
    # print(p.getLinkState(self.kinovaUid, self.kinovaEndEffectorIndex, computeForwardKinematics =1))
    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath,"tray/traybox.urdf"), 0.6600,0.000,-0.0000,0.000000,0.000000,1.000000,0.000000)  #0.44  0.075  -0.19
    self.endEffectorPos = [0.061,0.0,1.121]  #0.537,0.0,0.5
    # self.endEffectorPos = [0.0,0.0,0.0]  #0.537,0.0,0.5
    # (0.06106187295331401, -5.7852693706624846e-05, 1.121078907978537)
    self.endEffectorAngle = 0.0
    self.motorNames = []
    self.motorIndices = []
    
    for i in range (self.numJoints):
      jointInfo = p.getJointInfo(self.kinovaUid,i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6 #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.kinovaUid,self.kinovaGripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)
        
    observation.extend(list(pos))
    observation.extend(list(euler))
    
    return observation

  def applyAction(self, motorCommands, mode):
    
    #print ("self.numJoints")
    #print (self.numJoints)
    # print("applyaction:")
    # print(motorCommands)
    if (self.useInverseKinematics):
      
      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]
      
      state = p.getLinkState(self.kinovaUid,self.kinovaEndEffectorIndex)
      # for joint in range(p.getNumJoints(self.kinovaUid)):
      #   print("================")
      #   print(p.getJointInfo(self.kinovaUid,joint))

      actualEndEffectorPos = state[0]
      # print("pos[2] (getLinkState(kinovaEndEffectorIndex)")
      # print(actualEndEffectorPos)
      # exit()
    
      pos = [0, 0, 0]
      # self.endEffectorPos[0] = self.endEffectorPos[0]+dx
      # pos[0] = self.endEffectorPos[0]+dx
      pos[0] = dx
      # pos[0] = self.endEffectorPos[0]+dx - 0.061
      # if (self.endEffectorPos[0]>0.65):
      #   self.endEffectorPos[0]=0.65
      # if (self.endEffectorPos[0]<0.50):
      #   self.endEffectorPos[0]=0.50
      # self.endEffectorPos[1] = self.endEffectorPos[1]+dy
      pos[1] = dy

      # 0.061, 0.0, 1.121
      # pos[1] = self.endEffectorPos[1]+dy
      # if (self.endEffectorPos[1]<-0.17):
      #   self.endEffectorPos[1]=-0.17
      # if (self.endEffectorPos[1]>0.22):
      #   self.endEffectorPos[1]=0.22
      
      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      # self.endEffectorPos[2] = self.endEffectorPos[2]+dz
      pos[2] = dz

      # pos = motorCommands[:3]
      # pos[2] -= 0.05 
      # print(pos)
      # pos= [1.5,0,0.4]
      # self.endEffectorAngle = self.endEffectorAngle + da
      endEffectorAngle = da
      # print("endEffectorAngle")
      # print(endEffectorAngle)
      # pos = self.endEffectorPos

      # print('+++++++++abs pose:+++++++++')
      # print(pos)
      # print('+++++++++abs angles:+++++++++')
      # print(endEffectorAngle)
      # pos = [0.46,0,0.8]

      if mode==1:
        # print("moving")
        # orn = p.getQuaternionFromEuler([0,math.pi,0]) # -math.pi,yaw])   0326
        orn = p.getQuaternionFromEuler([0,math.pi,-endEffectorAngle]) # -math.pi,yaw])

        if (self.useNullSpace==1):
          if (self.useOrientation==1):
            jointPoses = p.calculateInverseKinematics(self.kinovaUid,self.kinovaEndEffectorIndex,pos,orn,self.ll,self.ul,self.jr,self.rp)
          else:
            jointPoses = p.calculateInverseKinematics(self.kinovaUid,self.kinovaEndEffectorIndex,pos,lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
        else:
          if (self.useOrientation==1):
            # jointPoses = p.calculateInverseKinematics(self.kinovaUid,self.kinovaEndEffectorIndex,pos,orn,jointDamping=self.jd)
            jointPoses = p.calculateInverseKinematics(self.kinovaUid,self.kinovaEndEffectorIndex,pos,orn,jointDamping=self.jd)
            # print("this method")
          else:
            jointPoses = p.calculateInverseKinematics(self.kinovaUid,self.kinovaEndEffectorIndex,pos)

        # print("jointPoses")
        # print(len(jointPoses))
        # print("self.kinovaEndEffectorIndex")
        # print(self.kinovaEndEffectorIndex)
        # p.setJointMotorControl2(self.kinovaUid, 8, p.POSITION_CONTROL, targetPosition=endEffectorAngle,force=self.maxForce*0.5)

        if (self.useSimulation):
          # for i in range (self.kinovaEndEffectorIndex):
          for i in range (7):
            #print(i)
            p.setJointMotorControl2(bodyUniqueId=self.kinovaUid,jointIndex=i+2,controlMode=p.POSITION_CONTROL,targetPosition= jointPoses[i],targetVelocity=0,\
              force=self.maxForce*1,maxVelocity=self.maxVelocity, positionGain=0.03,velocityGain=1)
            # p.setJointMotorControl2(self._kinova,jointIndex=joint,controlMode=POSITION_CONTROL, targetPosition=0, force=2000)
            # pass
        else:
          #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
          for i in range (self.numJoints):
            p.resetJointState(self.kinovaUid,i,jointPoses[i])
        # p.setJointMotorControl2(self.kinovaUid,jointIndex=8,controlMode=p.VELOCITY_CONTROL, targetVelocity=10, force=20)



        #fingers
      else:
        # print("finger!")

        p.setJointMotorControl2(self.kinovaUid,10,p.POSITION_CONTROL,targetPosition=fingerAngle*1,force=self.fingerBForce)
        p.setJointMotorControl2(self.kinovaUid,12,p.POSITION_CONTROL,targetPosition=fingerAngle*1,force=self.fingerBForce)
        p.setJointMotorControl2(self.kinovaUid,14,p.POSITION_CONTROL,targetPosition=fingerAngle*1,force=self.fingerBForce)


        p.setJointMotorControl2(self.kinovaUid,11,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
        p.setJointMotorControl2(self.kinovaUid,13,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
        p.setJointMotorControl2(self.kinovaUid,15,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)

        # p.setJointMotorControl2(self.kinovaUid, 8, p.POSITION_CONTROL, targetPosition=endEffectorAngle,force=self.maxForce*0.5)

      # jointpos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      # for i in range(p.getNumJoints(self.kinovaUid)):
      #   jointpos[i] = p.getJointState(self.kinovaUid,i)[0]
      #
      #   # print(p)
      # print(jointpos)
    else:
      for action in range (len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kinovaUid,motor,p.POSITION_CONTROL,targetPosition=motorCommands[action],force=self.maxForce)
      
