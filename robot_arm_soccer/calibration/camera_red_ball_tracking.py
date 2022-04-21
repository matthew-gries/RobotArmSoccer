"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""
import cv2
import numpy as np

def show_webcam():
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()

        if not ret_val:
            continue

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
        # cv2.imshow('Object distance test', img_hsv)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()