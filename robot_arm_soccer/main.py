from nuro_arm.robot.robot_arm import RobotArm
import cv2
import numpy as np
import pathlib

from typing import Tuple

# folder this file is in
FILE_DIRECTORY = pathlib.Path(__file__).parent

# f = 3.6 mm
WEBCAM_FOCAL_LENGTH = 3.6

# Diameter of ball = 63.5 mm
BALL_DIAMETER = 63.5

# Width of image captured
PIXEL_WIDTH = 1440

# Height of image captured
PIXEL_HEIGHT = 960

# Distance cut-off, in mm
DISTANCE_CUTOFF = 200

# Arm joint positions for the "ready" position
NONBLOCKING_JPOS = [0.0041887902047863905, -1.0974630336540343, 1.3236577047124993, 1.4535102010608776, -0.0041887902047863905]

# Arm joint positions for blocking
BLOCKING_JPOS = [0.0041887902047863905, -0.3099704751541929, 1.7928022076485752, 1.126784565087539, -0.0041887902047863905]

def load_camera_mtx() -> np.ndarray:
    mtx = np.load(str(FILE_DIRECTORY / "calibration" / "parameters" / "mtx.npy"))
    return mtx

def pixels_per_mm(mtx: np.ndarray) -> Tuple[float, float]:
    f_x = mtx[0][0]
    f_y = mtx[1][1]

    m_x = f_x / WEBCAM_FOCAL_LENGTH
    m_y = f_y / WEBCAM_FOCAL_LENGTH

    return (m_x, m_y)

def main():
    robot = RobotArm()
    robot.move_arm_jpos(NONBLOCKING_JPOS)
    robot.close_gripper()

    cam = cv2.VideoCapture(0)
    mtx = load_camera_mtx()
    m_x, m_y = pixels_per_mm(mtx)

    blocking = False

    while True:
        ret_val, img = cam.read()

        if not ret_val:
            continue

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img_hsv.copy()

        mask1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([8, 255, 255]))
        mask2 = cv2.inRange(img_hsv, np.array([172, 100, 100]), np.array([180, 255, 255]))
        mask = np.bitwise_or(mask1, mask2)
        result = cv2.bitwise_and(result, result, mask=mask)

        # try to find contours to find and outline and ball
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            best_contour = max(contours, key=lambda x: cv2.contourArea(x))
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)

            obj_dist = (BALL_DIAMETER * WEBCAM_FOCAL_LENGTH) / (w / m_x)
            x_pixel_from_center_line = (PIXEL_WIDTH/2 - (x + w/2) + 200)
            x_mm_from_center_line = x_pixel_from_center_line * WEBCAM_FOCAL_LENGTH / m_x
            angle = np.arcsin(x_mm_from_center_line / obj_dist)

            print(f"Estimated object distance: {obj_dist} mm")
            if obj_dist < DISTANCE_CUTOFF and not blocking:
                blocking = True
                robot.move_arm_jpos(BLOCKING_JPOS)
            
            if obj_dist > DISTANCE_CUTOFF and blocking:
                blocking = False
                robot.move_arm_jpos(NONBLOCKING_JPOS)

        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()