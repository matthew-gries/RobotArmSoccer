"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""
import cv2
import numpy as np
from nuro_arm.robot.robot_arm import RobotArm

# Arm joint positions for the "ready" position
NONBLOCKING_JPOS = [0.0041887902047863905, -1.0974630336540343, 1.3236577047124993, 1.4535102010608776, -0.0041887902047863905]

def show_webcam():

    robot = RobotArm()
    robot.move_arm_jpos(NONBLOCKING_JPOS)
    robot.close_gripper()
    robot.passive_mode()
    cam = cv2.VideoCapture(1)

    while True:
        ret_val, img = cam.read()

        cv2.imshow('img', img)

        print(robot.get_hand_pose())
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()