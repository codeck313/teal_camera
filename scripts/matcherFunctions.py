
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from config import *


def filterImage(img):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=255)
    return out_gray


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

    aporad = int(min(image.shape) * 0.10)
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


def get_lograd(shape, log_base):
    ret = np.zeros(shape, dtype=np.float64)
    ret += np.power(log_base, np.arange(shape[1], dtype=float))[np.newaxis, :]
    return ret


def wrap_angle(angles, ceil=2 * np.pi):
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def get_constraint_mask(shape, log_base, constraints=None):
    if constraints is None:
        constraints = {}

    mask = np.ones(shape, float)

    if "scale" in constraints:
        scale, sigma = constraints["scale"]
        scales = np.fft.ifftshift(get_lograd(shape, log_base))
        scales *= log_base ** (- shape[1] / 2.0)
        scales -= 1.0 / scale
        if sigma == 0:
            ascales = np.abs(scales)
            scale_min = ascales.min()
            mask[ascales > scale_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-scales ** 2 / sigma ** 2)

    if "angle" in constraints:
        angle, sigma = constraints["angle"]
        angles = -np.linspace(0, np.pi,
                              shape[0], endpoint=False)[:, np.newaxis]
        angles += np.deg2rad(angle)
        wrap_angle(angles, np.pi)
        angles = np.rad2deg(angles)
        if sigma == 0:
            aangles = np.abs(angles)
            angle_min = aangles.min()
            mask[aangles > angle_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-angles ** 2 / sigma ** 2)

    mask = np.fft.fftshift(mask)
    return mask


def _argmax2D(array, reports=None):
    amax = np.argmax(array)
    ret = list(np.unravel_index(amax, array.shape))
    return np.array(ret)


def get_subarr(array, center, rad):
    dim = 1 + 2 * rad
    subarr = np.zeros((dim,) * 2)
    corner = np.array(center) - rad
    for ii in range(dim):
        yidx = corner[0] + ii
        yidx %= array.shape[0]
        for jj in range(dim):
            xidx = corner[1] + jj
            xidx %= array.shape[1]
            subarr[ii, jj] = array[yidx, xidx]
    return subarr


def interpolate(array, rough, rad=2):
    rough = np.round(rough).astype(int)
    surroundings = get_subarr(array, rough, rad)
    com = argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def argmax_ext(array, exponent):
    # COM thing
    ret = None
    if exponent == "inf":
        ret = _argmax2D(array)
    else:
        col = np.arange(array.shape[0])[:, np.newaxis]
        row = np.arange(array.shape[1])[np.newaxis, :]

        arr2 = array ** exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return np.zeros(2)
        arrprody = np.sum(arr2 * col) / arrsum
        arrprodx = np.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return np.array(ret)


def success_get(array, coord, radius=2):
    coord = np.round(coord).astype(int)
    coord = tuple(coord)

    subarr = get_subarr(array, coord, 2)

    theval = subarr.sum()
    theval2 = array[coord]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    success = np.sqrt(theval * theval2)
    return success


def argmax_angscale(array, log_base, exponent, constraints=None, reports=None):
    mask = get_constraint_mask(array.shape, log_base, constraints)
    array_orig = array.copy()

    array *= mask
    ret = argmax_ext(array, exponent)
    ret_final = interpolate(array, ret)

    if reports is not None and reports.show("scale_angle"):
        reports["amas-orig"] = array_orig.copy()
        reports["amas-postproc"] = array.copy()

    success = success_get(array_orig, tuple(ret_final), 0)
    return ret_final, success


def phase_correlation(im0, im1, callback=None, *args):

    f0, f1 = [np.fft.fft2(arr) for arr in (im0, im1)]
    eps = abs(f1).max() * 1e-15
    cps = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    scps = np.fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def calc_phase_correlation(img1, img2, log_base, pcorr_shape):
    (arg_ang, arg_rad), success = phase_correlation(
        img1, img2, argmax_angscale, log_base, 'inf', None, None)

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = wrap_angle(angle, 360)
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


def templateMatch(img, template, angle, fftSuc):
    template = rotate_image(template, angle)
    # cv2.imwrite(
    #     "/home/redop/catkin_ws/src/teal_camera/scripts/Template.png", template)
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
    print(cen[0], cen[1], angle)
    if (max_val >= THRESHOLD) and (fftSuc >= THRESHOLD_FFT) and (EQUAL_SCALE == False):
        print("Object Found")
        return img, cen[0], cen[1], angle
    if (fftSuc >= THRESHOLD_FFT) and (EQUAL_SCALE == True):
        print("Object Found")
        return img, (cen[0]*100.0/scaleAngleMatcher), (cen[1] * 100.0/scaleAngleMatcher), angle
    return img, -1, -1, 0
