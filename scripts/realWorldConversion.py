import cv2
import numpy as np
cameraMatrix = np.array([[4.58118220e+03, 0.00000000e+00, 1.19881128e+03],
                         [0.00000000e+00, 4.57898831e+03, 9.10908996e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortionCoefficient = np.array([[-5.21104196e-01,  3.04887219e+00,
                                   3.54839033e-03,  4.36924889e-03,  - 1.47829244e+01]])


def Image_undistortion(input_image):
    h,  w = input_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distortionCoefficient, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(input_image, cameraMatrix,
                        distortionCoefficient, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst


def position_calculator(current_x, current_y, origin_pix, coefficient):
    current_x_real = (origin_pix[0]-current_x)/coefficient[0]
    current_y_real = (origin_pix[1]-current_y)/coefficient[1]
    return current_x_real, current_y_real


def coefficient_calculator(origin_pix, Known_pix, realWorld):
    coefficient_x = (origin_pix[0]-Known_pix[0])/realWorld[0]
    coefficient_y = (origin_pix[1]-Known_pix[1])/realWorld[1]
    return (coefficient_x, coefficient_y)
