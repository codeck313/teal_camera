
import matplotlib.pyplot as plt
import cv2
import numpy as np
from linetimer import CodeTimer
import imutils
import imreg_dft
from config import *


def filterImage(img):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=255)
    return img


def get_borderval(img, radius=None):
    """
    Given an image and a radius, examine the average value of the image
    at most radius pixels from the edge
    """
    if radius is None:
        mindim = min(img.shape)
        radius = max(1, mindim // 20)
    mask = np.zeros_like(img, dtype=np.bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[:radius, :] = True
    mask[-radius:, :] = True
    mean = np.median(img[mask])
    return mean


def highPass(shape):
    # TODO
    """
    radial cosine filter, suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy ** 2 + xx ** 2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    return filt


def logpolar(image, sqaure_shape, log_base):
    """Return log-polar transformed image and log base."""

    # imshape = np.array(image.shape)
    center = image.shape[0] / 2.0, image.shape[1] / 2.0

    # going to consider a sqauare image so need to compensate for that using this ratio
    aspect_ratio = image.shape[0]/float(image.shape[1])

    # log base from Fourier Mellin transform
    scale = np.power(log_base,
                     np.arange(sqaure_shape[1], dtype=np.float32))[np.newaxis, :]
    angle = -np.linspace(0, np.pi, sqaure_shape[0], endpoint=False,
                         dtype=np.float32)[:, np.newaxis]

    yMap = scale*np.sin(angle) + center[0]
    xMap = scale*(np.cos(angle)/aspect_ratio) + center[1]
    output = cv2.remap(image, xMap, yMap, cv2.INTER_CUBIC)
    return output


def get_apofield(shape, aporad):
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield


def apodize(image):

    aporad = int(min(image.shape) * 0.12)
    apofield = get_apofield(image.shape, aporad)
    res = image * apofield
    bg = get_borderval(image, aporad // 2)
    res += bg * (1 - apofield)
    return res


def fft_log_polar(imgs):
    imgs = [apodize(img) for img in imgs]  # about 0.006 diff
    shape = imgs[0].shape
    dfts = [np.fft.fftshift(np.fft.fft2(img)) for img in imgs]
    filt = highPass(shape)
    dfts = [dft * filt for dft in dfts]
    square_shape = (max(shape),)*2
    log_base = np.exp(np.log(shape[0]*1.1/2.0) / max(shape))
    stuffs = [logpolar(np.abs(dft), square_shape, log_base)
              for dft in dfts]
    return stuffs, log_base, square_shape


def calc_phase_correlation(img1, img2, log_base, pcorr_shape):
    (arg_ang, arg_rad), success = imreg_dft.imreg._phase_correlation(
        img1, img2,
        imreg_dft.utils.argmax_angscale, log_base, 'inf', None, None)

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = imreg_dft.utils.wrap_angle(angle, 360)
    scale = log_base ** arg_rad

    angle = - angle
    scale = 1.0 / scale
    print("Angle and Success")
    print(angle, success)
    # float 64, float 64
    return angle, success


def rotate_image(image, angle):
    rotated_image = imutils.rotate_bound(image, angle)
    return rotated_image


def templateMatch(img, template, angle):
    template = rotate_image(template, angle)
    cv2.imwrite(
        "/home/redop/catkin_ws/src/teal_camera/scripts/Template.png", template)
    w, h = template.shape[::-1]
    # with CodeTimer():
    res = cv2.matchTemplate(img, template, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 0, 2)
    cen = (top_left[0]+w/2, top_left[1]+h/2)
    # TODO can remove
    cv2.circle(img, cen, 2, 0, 2)
    print("Max, Min, Max_val")
    print(max_loc, min_loc, max_val)
    # if cv2.waitKey(0) == 32:
    #     cv2.imshow("result", img)
    return img, cen[0], cen[1], angle
