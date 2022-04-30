"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""
import cv2
import numpy as np
from nuro_arm.robot.robot_arm import RobotArm

import pathlib

# folder this file is in
FILE_DIRECTORY = pathlib.Path(__file__).parent

# Width of image captured
PIXEL_WIDTH = 1440

# Height of image captured
PIXEL_HEIGHT = 960

# Arm joint positions for the "ready" position
NONBLOCKING_JPOS = [0.0041887902047863905, -1.0974630336540343, 1.3236577047124993, 1.4535102010608776, -0.0041887902047863905]

def load_camera_mtx() -> np.ndarray:
    mtx = np.load(str(FILE_DIRECTORY / "parameters" / "mtx.npy"))
    return mtx

def load_dist_mtx() -> np.ndarray:
    dist = np.load(str(FILE_DIRECTORY / "parameters" / "dist.npy"))
    return dist

def show_webcam():

    robot = RobotArm()
    robot.move_arm_jpos(NONBLOCKING_JPOS)
    robot.close_gripper()
    robot.passive_mode()
    cam = cv2.VideoCapture(1)
    mtx = load_camera_mtx()
    dist = load_dist_mtx()

    while True:
        ret_val, img = cam.read()
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (PIXEL_WIDTH, PIXEL_HEIGHT), 1, (PIXEL_WIDTH, PIXEL_HEIGHT))
        dst = cv2.undistort(img, mtx, dist, None, new_cam_mtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv2.imshow('img', dst)

        print(robot.get_hand_pose())
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()