import numpy as np
import cv2
from typing import Tuple
import pathlib

# folder this file is in
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

def load_camera_mtx() -> np.ndarray:
    mtx = np.load(str(FILE_DIRECTORY / "parameters" / "mtx.npy"))
    return mtx

def pixels_per_mm(mtx: np.ndarray) -> Tuple[float, float]:
    f_x = mtx[0][0]
    f_y = mtx[1][1]

    m_x = f_x / WEBCAM_FOCAL_LENGTH
    m_y = f_y / WEBCAM_FOCAL_LENGTH

    return (m_x, m_y)

def show_webcam():
    cam = cv2.VideoCapture(1)
    mtx = load_camera_mtx()
    m_x, m_y = pixels_per_mm(mtx)

    while True:
        ret_val, img = cam.read()

        if not ret_val:
            continue

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img_hsv.copy()

        # mask1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([8, 255, 255]))
        # mask2 = cv2.inRange(img_hsv, np.array([172, 100, 100]), np.array([180, 255, 255]))
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
            x_pixel_from_center_line = (PIXEL_WIDTH/2 - (x + w/2) + 200)
            x_mm_from_center_line = x_pixel_from_center_line * WEBCAM_FOCAL_LENGTH / m_x
            angle = np.arcsin(x_mm_from_center_line / obj_dist)

            print(f"Estimated object distance: {obj_dist} mm")
            # print(f"Estimated angle from center: {angle} rad")
            # print(x, y, w, h)
            # print(x_pixel_from_center_line)
            # print(x_mm_from_center_line)

        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()

