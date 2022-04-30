from nuro_arm.robot.robot_arm import RobotArm
import cv2
import numpy as np
import pathlib
import time

from typing import Tuple

# Folder this file is in
FILE_DIRECTORY = pathlib.Path(__file__).parent

# f = 3.6 mm
WEBCAM_FOCAL_LENGTH = 3.6

# Diameter of ball = 63.5 mm
# BALL_DIAMETER = 63.5
BALL_DIAMETER = 38.09

# Width of image captured
PIXEL_WIDTH = 1440

# Height of image captured
PIXEL_HEIGHT = 960

# Distance cut-off, in mm from camera lens
DISTANCE_CUTOFF = 500

# Distance of the camera lens to the x-position aligned with the gripper, in mm from camera lens
CAM_2_GRIPPER_X = 100

# Arm joint positions for the "ready" position
NONBLOCKING_JPOS = [0.0041887902047863905, -0.4775220833456485, 1.5917402778188283, 1.2692034320502763, -0.0041887902047863905]

# Countdown before starting, in seconds
COUNTDOWN = 5

# Approximate FPS of webcam
APPROX_FPS = 60

# Hyperparameter to estimate the amount of time the roll will be (better to over-estimate than under-estimate), in seconds
ESTIMATED_ROLL_TIME = 20

# The distance, in mm, of the goal line in the FOV of the camera
GOAL_LINE_IN_FOV = 130

# The distance, in mm, from the robot base to where the camera FOV begins on the side of the robot
CAM_2_FOV_DISPLACEMENT = 76.2 # FOV starts from the edge of the robot base

def load_camera_mtx() -> np.ndarray:
    mtx = np.load(str(FILE_DIRECTORY / "calibration" / "parameters" / "mtx.npy"))
    return mtx

def load_dist_mtx() -> np.ndarray:
    dist = np.load(str(FILE_DIRECTORY / "calibration" / "parameters" / "dist.npy"))
    return dist

def pixels_per_mm(mtx: np.ndarray) -> Tuple[float, float]:
    f_x = mtx[0][0]
    f_y = mtx[1][1]

    m_x = f_x / WEBCAM_FOCAL_LENGTH
    m_y = f_y / WEBCAM_FOCAL_LENGTH

    return (m_x, m_y)

def get_gripper_coord(x: float, y: float) -> Tuple[float, float]:
    # IGNORE APPROXIMATING THE Z POSITION, just hardcode for now
    # robot_pos_per_pixel = (PIXEL_ROBOT_POS_MAP['max_x'] - PIXEL_ROBOT_POS_MAP['min_x']) / PIXEL_WIDTH
    # x_pos_robot = x * robot_pos_per_pixel
    # return (x_pos_robot, 0.25)

    avg_mm_per_px = GOAL_LINE_IN_FOV / PIXEL_WIDTH
    print(avg_mm_per_px)
    x_center_in_mm = x * avg_mm_per_px
    print(x_center_in_mm)
    x_center_from_robot_base = x_center_in_mm + CAM_2_FOV_DISPLACEMENT
    print(x_center_from_robot_base)
    # 0.1 in robot coordinates is approx 25.4 mm
    robot_x_coord = (x_center_from_robot_base / 25.4) * 0.025
    return (robot_x_coord, 0.025)



def main():
    robot = RobotArm()
    robot.move_arm_jpos(NONBLOCKING_JPOS)
    robot.close_gripper()

    cam = cv2.VideoCapture(1)
    mtx = load_camera_mtx()
    dist = load_dist_mtx()
    m_x, m_y = pixels_per_mm(mtx)

    # Pre-allocate a numpy array to keep track of the distance (in mm) of the ball from the camera at each time
    # step (this is the straight line y-axis distance), row 1 is the time captured and row 2 is the distance
    dist_array = np.zeros((2, APPROX_FPS * ESTIMATED_ROLL_TIME))

    # Pre-allocate a numpy array to keep track of the center of the ball along the x-axis of image, in pixels, row 1 is the
    # time captured and row 2 is the distance
    x_center_array = np.zeros((2, APPROX_FPS * ESTIMATED_ROLL_TIME))

    # Pre-allocate a numpy array to keep track of the center of the ball along the y-axis of image, in pixels, row 1 is the
    # time captured and row 2 is the distance
    y_center_array = np.zeros((2, APPROX_FPS * ESTIMATED_ROLL_TIME))

    frame_index = 0

    print("Press ENTER when ready to start the counter!")
    input()

    print(f"Starting in {COUNTDOWN} seconds!...")
    start = time.time()
    while time.time() - start < COUNTDOWN:
        pass

    print("Go!")

    start = time.time()
    while True:
        ret_val, img = cam.read()
        # get time stamp of this frame
        t = time.time() - start

        if not ret_val:
            continue

        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (PIXEL_WIDTH, PIXEL_HEIGHT), 1, (PIXEL_WIDTH, PIXEL_HEIGHT))
        undistored = cv2.undistort(img, mtx, dist, None, new_cam_mtx)
        # crop the image
        x, y, w, h = roi
        img = undistored[y:y+h, x:x+w]

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img_hsv.copy()

        # mask1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([8, 255, 255]))
        # mask2 = cv2.inRange(img_hsv, np.array([172, 100, 100]), np.array([180, 255, 255]))
        # mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
        # mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
        mask1 = cv2.inRange(img_hsv, (0,50,20), (10,255,255))
        mask2 = cv2.inRange(img_hsv, (170,50,20), (180,255,255))
        mask = np.bitwise_or(mask1, mask2)
        result = cv2.bitwise_and(result, result, mask=mask)

        # try to find contours to find and outline and ball
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            best_contour = max(contours, key=lambda x: cv2.contourArea(x))
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)

            obj_dist = (BALL_DIAMETER * WEBCAM_FOCAL_LENGTH) / (w / m_x)

            print("Time: {:.2f}\tEst. dist: {:.3f} mm\tCenter: ({:.3f}, {:.3f}) px".format(t, obj_dist, x+w/2, y+h/2))

            if frame_index < APPROX_FPS * ESTIMATED_ROLL_TIME:
                dist_array[0,frame_index] = t
                dist_array[1,frame_index] = obj_dist
                x_center_array[0,frame_index] = t
                x_center_array[1,frame_index] = x + w/2
                y_center_array[0,frame_index] = t
                y_center_array[1,frame_index] = y + h/2
            else:
                dist_array = np.concatenate((dist_array, np.array([[t], [obj_dist]])))
                x_center_array = np.concatenate((x_center_array, np.array([[t], [x+w/2]])))
                y_center_array = np.concatenate((y_center_array, np.array([[t], [y+h/2]])))

            if obj_dist < DISTANCE_CUTOFF:
                print(f"Object passed threshold at {obj_dist} mm!")
                break

            frame_index += 1

        cv2.imshow('mask', mask)
        cv2.imshow('result', result)

    # if we did not use the whole buffer, remove ends of the buffers we did not use
    if frame_index < APPROX_FPS * ESTIMATED_ROLL_TIME:
        dist_array = dist_array[:,:frame_index]
        x_center_array = x_center_array[:,:frame_index]

    # Do least squares method on the pixels to approximate where the center of the ball will end up
    # in terms of pixels
    x_m, x_b = np.polyfit(x=x_center_array[0,:], y=x_center_array[1,:], deg=1)
    y_m, y_b = np.polyfit(x=y_center_array[0,:], y=y_center_array[1,:], deg=1)

    # Do least squares method on distance and time, approximates distance of ball (mm) from camera lens at
    # a given time
    dist_m, dist_b = np.polyfit(x=dist_array[0,:], y=dist_array[1,:], deg=1)

    # Get time step ball will be in front of gripper
    t_gripper = (CAM_2_GRIPPER_X - dist_b) / dist_m

    print(f"GRIPPER TIME: {t_gripper}")

    # Get the approximate image xy-position of the center of the ball when in front of the gripper, in pixels
    x_center_at_t_gripper = (x_m * t_gripper) + x_b
    y_center_at_t_gripper = (y_m * t_gripper) + y_b

    print(f"X CENTER AT T GRIPPER: {x_center_at_t_gripper}")

    # Get the x and z coordinates the robot should move its gripper to
    x_gripper, z_gripper = get_gripper_coord(x_center_at_t_gripper, y_center_at_t_gripper)

    success = robot.move_hand_to([x_gripper, 0, z_gripper])

    if not success:
        print("WARNING: Robot did not report a successful move")

    print(f"LEAST SQUARES APPROX. DIST:\tdist = {dist_m}t + {dist_b}")
    print(f"LEAST SQUARES APPROX. X CENTER:\tx_center = {x_m}t + {x_b}")
    print(f"LEAST SQUARES APPROX. Y CENTER:\ty_center = {y_m}t + {y_b}")
    print(f"GRIPPER MOVED TO: [{x_gripper} 0.0 {z_gripper}]")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()