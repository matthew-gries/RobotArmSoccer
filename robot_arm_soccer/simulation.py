from typing import Tuple, List, Optional, Dict
import time
import numpy as np
import pybullet as pb
import pybullet_data

class RobotArm:
    GRIPPER_CLOSED = 0.
    GRIPPER_OPENED = 1.
    def __init__(self):
        '''Robot Arm simulated in Pybullet, with support for performing top-down
        grasps within a specified workspace
        '''
        # placing robot higher above ground improves top-down grasping ability
        self._id = pb.loadURDF("assets/urdf/xarm.urdf",
                               basePosition=(0, 0, 0.05),
                               flags=pb.URDF_USE_SELF_COLLISION)

        # these are hard coded based on how urdf is written
        self.arm_joint_ids = [1,2,3,4,5]
        self.gripper_joint_ids = [6,7]
        self.dummy_joint_ids = [8]
        self.finger_joint_ids = [9,10]
        self.end_effector_link_index = 11

        self.arm_joint_limits = np.array(((-2, -1.58, -2, -1.8, -2),
                                          ( 2,  1.58,  2,  2.0,  2)))
        # self.gripper_joint_limits = np.array(((0.05,0.05),
        #                                       (1.38, 1.38)))
        # Don't open the gripper as much
        self.gripper_joint_limits = np.array(((0.075,0.075),
                                              (0.25, 0.25)))

        # chosen to move arm out of view of camera
        self.home_arm_jpos = [0., -1.1, 1.4, 1.3, 0.]

        # joint constraints are needed for four-bar linkage in xarm fingers
        for i in [0,1]:
            constraint = pb.createConstraint(self._id,
                                             self.gripper_joint_ids[i],
                                             self._id,
                                             self.finger_joint_ids[i],
                                             pb.JOINT_POINT2POINT,
                                             (0,0,0),
                                             (0,0,0.03),
                                             (0,0,0))
            pb.changeConstraint(constraint, maxForce=1000000)

        # reset joints in hand so that constraints are satisfied
        hand_joint_ids = self.gripper_joint_ids + self.dummy_joint_ids + self.finger_joint_ids
        hand_rest_states = [0.05, 0.05, 0.055, 0.0155, 0.031]
        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(hand_joint_ids, hand_rest_states)]

        # allow finger and linkages to move freely
        pb.setJointMotorControlArray(self._id,
                                     self.dummy_joint_ids+self.finger_joint_ids,
                                     pb.POSITION_CONTROL,
                                     forces=[0,0,0])

    def move_gripper_to(self, position: List[float], theta: float):
        '''Commands motors to move end effector to desired position, oriented
        downwards with a rotation of theta about z-axis

        Parameters
        ----------
        position
            xyz position that end effector should move toward
        theta
            rotation (in radians) of the gripper about the z-axis.

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        quat = pb.getQuaternionFromEuler((0,-np.pi,theta))
        arm_jpos, _ = self.solve_ik(position, quat)

        return self.move_arm_to_jpos(arm_jpos)

    def solve_ik(self,
                 pos: List[float],
                 quat: Optional[List[float]]=None,
                ) -> Tuple[List[float], Dict[str, float]]:
        '''Calculates inverse kinematics solution for a desired end effector
        position and (optionally) orientation, and returns residuals

        Hint
        ----
        To calculate residuals, you can get the pose of the end effector link using
        `pybullet.getLinkState` (but you need to set the arm joint positions first)

        Parameters
        ----------
        pos
            target xyz position of end effector
        quat
            target orientation of end effector as unit quaternion if specified.
            otherwise, ik solution ignores final orientation

        Returns
        -------
        list
            joint positions of arm that would result in desired end effector
            position and orientation. in order from base to wrist
        dict
            position and orientation residuals:
                {'position' : || pos - achieved_pos ||,
                 'orientation' : 1 - |<quat, achieved_quat>|}
        '''
        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 pos,
                                                 quat,
                                                 maxNumIterations=20,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]

        # teleport arm to check acheived pos and orientation
        old_arm_jpos = list(zip(*pb.getJointStates(self._id, self.arm_joint_ids)))[0]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, arm_jpos)]
        achieved_pos, achieved_quat = pb.getLinkState(self._id, self.end_effector_link_index)[:2]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, old_arm_jpos)]

        residuals = {'position' : np.linalg.norm(np.subtract(pos, achieved_pos)),
                     'orientation' : 1 - np.abs(np.dot(quat, achieved_quat))}

        return arm_jpos, residuals

    def move_arm_to_jpos(self, arm_jpos: List[float]) -> bool:
        '''Commands motors to move arm to desired joint positions

        Parameters
        ----------
        arm_jpos
            joint positions (radians) of arm joints, ordered from base to wrist

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        # cannot use setJointMotorControlArray because API does not expose
        # maxVelocity argument, which is needed for stable object manipulation
        for j_id, jpos in zip(self.arm_joint_ids, arm_jpos):
            pb.setJointMotorControl2(self._id,
                                     j_id,
                                     pb.POSITION_CONTROL,
                                     jpos,
                                     positionGain=0.2,
                                     maxVelocity=0.8)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

    def set_gripper_state(self, gripper_state: float) -> bool:
        '''Commands motors to move gripper to given state

        Parameters
        ----------
        gripper_state
            gripper state is a continuous number from 0. (fully closed)
            to 1. (fully open)

        Returns
        -------
        bool
            True if movement is successful, False otherwise.

        Raises
        ------
        AssertionError
            If `gripper_state` is outside the range [0,1]
        '''
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]

        pb.setJointMotorControlArray(self._id,
                                     self.gripper_joint_ids,
                                     pb.POSITION_CONTROL,
                                     gripper_jpos,
                                     positionGains=[0.2, 0.2])

        success = self.monitor_movement(gripper_jpos, self.gripper_joint_ids)
        return success

    def monitor_movement(self,
                         target_jpos: List[float],
                         joint_ids: List[int],
                        ) -> bool:
        '''Monitors movement of motors to detect early stoppage or success.

        Note
        ----
        Current implementation calls `pybullet.stepSimulation`, without which the
        simulator will not move the motors.  You can avoid this by setting
        `pybullet.setRealTimeSimulation(True)` but this is usually not advised.

        Parameters
        ----------
        target_jpos
            final joint positions that motors are moving toward
        joint_ids
            the joint ids associated with each `target_jpos`, used to read out
            the joint state during movement

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        old_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
        while True:
            [pb.stepSimulation() for _ in range(10)]

            time.sleep(0.01)

            achieved_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
            if np.allclose(target_jpos, achieved_jpos, atol=1e-3):
                # success
                return True

            if np.allclose(achieved_jpos, old_jpos, atol=1e-3):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class Camera:
    def __init__(self, workspace: np.ndarray) -> None:
        '''Camera that is mounted to view workspace from above

        Hint
        ----
        For this camera setup, it may be easiest if you use the functions
        `pybullet.computeViewMatrix` and `pybullet.computeProjectionMatrixFOV`.
        cameraUpVector should be (0,1,0)

        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))

        Attributes
        ----------
        img_width : int
            width of rendered image
        img_height : int
            height of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_width = 100
        self.img_height = 100

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, 0.25)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(0,1,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=25,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)

    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image

        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_width,
                                 height=self.img_height,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]


class GraspingEnv:
    def __init__(self, render: bool=True) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        pb.setPhysicsEngineParameter(numSubSteps=0,
                                     numSolverIterations=100,
                                     solverResidualThreshold=1e-7,
                                     constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
        pb.setGravity(0,0,-10)

        # create ground plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # offset plane y-dim to place white tile under workspace
        self.plane_id = pb.loadURDF('plane.urdf', (0,-0.5,0))

        # makes collisions with plane more stable
        pb.changeDynamics(self.plane_id, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        # add robots
        self.goalie_robot = RobotArm()

        # move goalie over a bit and rotate PI radians
        pb.resetBasePositionAndOrientation(
            bodyUniqueId=self.striker_robot._id,
            posObj=[1.0, 0.0, 0.05],
            ornObj=pb.getQuaternionFromEuler([0, 0, np.pi])
        )

        # add the ball
        self.ball_id = self.create_ball(
            radius=0.025, 
            start_pos=[0.5, 0.0, 0.05], 
            start_orn_euler=[0.0, 0.0, 0.0]
        )

        self.workspace = np.array(((0.10, -0.12), # ((min_x, min_y)
                                   (0.10, 0.12))) #  (max_x, max_y))

        self.grasp_height = 0.1

        if render:
            self.draw_workspace()

        # add camera
        self.camera = Camera(self.workspace)


    @staticmethod
    def create_ball(radius: float, start_pos: List[float], start_orn_euler: List[float]) -> Tuple[int, int, int]:
        '''
        Create the ball to use in the simulation. Also sets the initial position and orientation (in radians)

        Returns a tuple representing (object_id, collision_id, visual_id)
        '''

        # create collision item
        coll_id = pb.createCollisionShape(
            shapeType=pb.GEOM_SPHERE,
            radius=radius
        )

        # create visual item
        vis_id = pb.createVisualShape(
            shapeType=pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1.0, 0.0, 0.0, 1.0]
        )

        object_id = pb.createMultiBody(0, coll_id, vis_id, basePosition=start_pos, baseOrientation=pb.getQuaternionFromEuler(start_orn_euler))

        pb.changeDynamics(
            object_id, 
            -1,
            lateralFriction=1,
            spinningFriction=0.005,
            rollingFriction=0.005
        )

        return (object_id, coll_id, vis_id)

    def draw_workspace(self) -> None:
        '''This is just for visualization purposes, to help you with the object
        resetting.  Must be in GUI mode, otherwise error occurs

        Note
        ----
        Pybullet debug lines only show up in GUI mode so they won't help you
        with camera placement.
        '''
        corner_ids = ((0,0), (0,1), (1,1), (1,0), (0,0))
        for i in range(4):
            start = (*self.workspace[corner_ids[i],[0,1]], 0.)
            end = (*self.workspace[corner_ids[i+1],[0,1]], 0.)
            pb.addUserDebugLine(start, end, (0,0,0), 3)

    def perform_grasp(self, x, y, theta) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)

        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta
            target rotation about z-axis of gripper during grasp

        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        robot = self.goalie_robot
        robot.move_arm_to_jpos(robot.home_arm_jpos)
        robot.set_gripper_state(robot.GRIPPER_OPENED)
    
        pos = np.array((x, y, self.grasp_height))
        robot.move_gripper_to(pos, theta)
        robot.set_gripper_state(robot.GRIPPER_CLOSED)

        robot.move_arm_to_jpos(robot.home_arm_jpos)

        # TODO figure out how to check for success
        return True

    def take_picture(self) -> np.ndarray:
        '''Takes picture using camera

        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()

def test_env():
    '''Test the the evironment is set up correctly
    '''
    env = GraspingEnv(True)

    while 1:
        env.take_picture()
        env.perform_grasp(0.02, 0.0, 0.0)


if __name__ == "__main__":
    test_env()
