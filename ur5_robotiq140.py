import os
import time
import pdb
import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt

serverMode = p.GUI # GUI/DIRECT
sisbotUrdfPath = "./urdf/ur5_robotiq_140.urdf"

# connect to engine servers
physicsClient = p.connect(serverMode)
# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#p.getCameraImage(640,480)

# define world
#p.setGravity(0,0,-10) # NOTE
planeID = p.loadURDF("plane.urdf")

# define environment
deskStartPos = [0.1, -0.49, 0.85]
deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("./urdf/objects/block.urdf", deskStartPos, deskStartOrientation)

tableStartPos = [0.0, -0.9, 0.8]
tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)

ur5standStartPos = [-0.7, -0.36, 0.0]
ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId1 = p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation,useFixedBase = True)

# define camera image parameter

width = 128
height = 128
fov = 40
aspect = width / height
near = 0.2
far = 2
view_matrix = p.computeViewMatrix([0.0, 1.5, 0.5], [0, 0, 0.7], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


# setup ur5 with robotiq 140
robotStartPos = [0,0,0.0]
robotStartOrn = p.getQuaternionFromEuler([0,0,0])
print("----------------------------------------")
print("Loading robot from {}".format(sisbotUrdfPath))
robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,
                     flags=p.URDF_USE_INERTIA_FROM_FILE)
joints, controlRobotiqC2, controlJoints, mimicParentName = utils_ur5_robotiq140.setup_sisbot(p, robotID)
eefID = 7 # ee_link

# start simulation
ABSE = lambda a,b: abs(a-b)

# set damping for robot arm and gripper
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
jd = jd*0

userParams = dict()

try:
    flag = True
    # custom sliders to tune parameters (name of the parameter,range,initial value)
    # Task space (Cartesian space)
    xin = p.addUserDebugParameter("x", -3.14, 3.14, 0.11)
    yin = p.addUserDebugParameter("y", -3.14, 3.14, -0.49)
    zin = p.addUserDebugParameter("z", 0.9, 1.3, 1.29)
    rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0) #-1.57 yaw
    pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 1.57)
    yawId = p.addUserDebugParameter("yaw", -3.14, 3.14, -1.57) # -3.14 pitch
    gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",0,0.085,0.085)

    # Joint space 
    userParams[0] = p.addUserDebugParameter("shoulder_pan_joint", -3.14, 3.14, -1.57)
    userParams[1] = p.addUserDebugParameter("shoulder_lift_joint", -3.14, 3.14, -1.57)
    userParams[2] = p.addUserDebugParameter("elbow_joint", -3.14, 3.14, 1.57)
    userParams[3] = p.addUserDebugParameter("wrist_1_joint",-3.14, 3.14, -1.57)
    userParams[4] = p.addUserDebugParameter("wrist_2_joint", -3.14, 3.14, -1.57)
    userParams[5] = p.addUserDebugParameter("wrist_3_joint", -3.14, 3.14, 0)   

    # Camera parameter for computeViewMatrix (see the pybullet document)
    c1 = p.addUserDebugParameter("cc1", -3, 5.5, 0.132)
    c2 = p.addUserDebugParameter("cc2", -3, 5.5, -1.524)
    c3 = p.addUserDebugParameter("cc3", -3, 5.5, 1.205)
    c4 = p.addUserDebugParameter("cc4", -3, 5.5, 0.132)
    c5 = p.addUserDebugParameter("cc5", -3, 5.5, -0.539)
    c6 = p.addUserDebugParameter("cc6", -3, 5.5, 1.116)

    control_cnt = 0;
    while(flag):


        # Get depth values using the OpenGL renderer
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        images = p.getCameraImage(width,height,view_matrix,projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
        #plt.imshow(rgb_opengl)
        #plt.title('RGB image')
        #plt.pause(0.0001)

        # read the value of task parameter
        x = p.readUserDebugParameter(xin)
        y = p.readUserDebugParameter(yin)
        z = p.readUserDebugParameter(zin)
        roll = p.readUserDebugParameter(rollId)
        pitch = p.readUserDebugParameter(pitchId)
        yaw = p.readUserDebugParameter(yawId)
        orn = p.getQuaternionFromEuler([roll, pitch, yaw])

        # read the value of camera parameter
        cc1 = p.readUserDebugParameter(c1)
        cc2 = p.readUserDebugParameter(c2)
        cc3 = p.readUserDebugParameter(c3)
        cc4 = p.readUserDebugParameter(c4)
        cc5 = p.readUserDebugParameter(c5)
        cc6 = p.readUserDebugParameter(c6)
        view_matrix = p.computeViewMatrix([cc1, cc2, cc3], [cc4, cc5, cc6], [0, 1, 0])

        gripper_opening_length = p.readUserDebugParameter(gripper_opening_length_control)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation

        # apply IK
        jointPose = p.calculateInverseKinematics(robotID, eefID, [x,y,z],orn,jointDamping=jd)
        for i, name in enumerate(controlJoints):
    
            joint = joints[name]
            pose = jointPose[i]
            # read joint value
            if i != 6:
                pose1 = p.readUserDebugParameter(userParams[i])

            if name==mimicParentName:
                controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
            else:
                if control_cnt < 100:
                    # control robot joints
                    p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose1, force=joint.maxForce, 
                                        maxVelocity=joint.maxVelocity)
                else:
                    # control robot end-effector
                    p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce, 
                                        maxVelocity=joint.maxVelocity)
        control_cnt = control_cnt + 1
        rXYZ = p.getLinkState(robotID, eefID)[0] # real XYZ
        rxyzw = p.getLinkState(robotID, eefID)[1] # real rpy
        rroll, rpitch, ryaw = p.getEulerFromQuaternion(rxyzw)
        print("err_x= {:.2f}, err_y= {:.2f}, err_z= {:.2f}".format(*list(map(ABSE,[x,y,z],rXYZ))))
        print("err_r= {:.2f}, err_o= {:.2f}, err_y= {:.2f}".format(*list(map(ABSE,[roll,pitch,yaw],[rroll, rpitch, ryaw]))))
        print("x_= {:.2f}, y= {:.2f}, z= {:.2f}".format(rXYZ[0],rXYZ[1],rXYZ[2]))
        print("rroll_= {:.2f}, rpitch= {:.2f}, ryaw= {:.2f}".format(rroll,rpitch,ryaw))
        # current box coordinate
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        print(cubePos,cubeOrn)

        p.stepSimulation()
    p.disconnect()
except KeyError:
    p.disconnect()
