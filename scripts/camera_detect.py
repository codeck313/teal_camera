#! /usr/bin/env python

# from imreg_dft import imreg
# import matplotlib.pyplot as plt
# from linetimer import CodeTimer
# import imutils
# import imreg_dft
import numpy as np
import cv2
import camera_bridge
from matcherFunctions import fft_log_polar, calc_phase_correlation, templateMatch, filterImage
import rospy
from config import *

im1 = (cv2.imread(TEMPLATE_BASE_NAME+".png", 0))
template = (cv2.imread(TEMPLATE_BASE_NAME+"_cropped.png", 0))


if EQUAL_SCALE is True:
    width = int(template.shape[1] * scaleAngleMatcher / 100)
    height = int(template.shape[0] * scaleAngleMatcher / 100)
    template = cv2.resize(template, (width, height),
                          interpolation=cv2.INTER_AREA)

width = int(im1.shape[1] * scaleAngleMatcher / 100)
height = int(im1.shape[0] * scaleAngleMatcher / 100)
dimAngle = (width, height)

# resize image
im1 = cv2.resize(im1, dimAngle, interpolation=cv2.INTER_AREA)


def findCoordinate(im1, template, im2):
    # im2 = filterImage(im2)
    if EQUAL_SCALE is False:
        im2Template = np.copy(im2)
    im2 = cv2.resize(im2, dimAngle, interpolation=cv2.INTER_AREA)
    img_res, log_base, pcorr_shape = fft_log_polar([im1, im2])
    result = calc_phase_correlation(
        img_res[0], img_res[1], log_base, pcorr_shape)
    if EQUAL_SCALE is True:
        return templateMatch(im2, template, result[0], result[1])
    else:
        return templateMatch(im2Template, template, result[0], result[1])


def main():
    rospy.init_node('camera_server', anonymous=True)
    camera_bridge.image_converter(findCoordinate, im1, template)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
