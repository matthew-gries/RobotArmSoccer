import numpy as np
import cv2
from typing import Tuple

# f = 3.6 mm
WEBCAM_FOCAL_LENGTH = 3.6

# Diameter of ball = 63.5 mm
BALL_DIAMETER = 63.5

def load_camera_mtx() -> np.ndarray:
    mtx = np.load("./parameters/mtx.npy")
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
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = img_hsv.copy()

        lower = np.array([155,25,0])
        upper = np.array([179,255,255])

        mask = cv2.inRange(img_hsv, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)

        # try to find contours to find and outline and ball
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)

        obj_dist = (BALL_DIAMETER * WEBCAM_FOCAL_LENGTH) / (w / m_x)

        print(f"Estimated object distance: {obj_dist} mm")

        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()

