from ctypes import *
import numpy as np
import os, os.path

import cv2
import time

package_directory = os.path.dirname(os.path.abspath(__file__))

ext_path = os.path.join(package_directory, "libcppext.so")
ext_dll = cdll.LoadLibrary(ext_path)


def non_extrema_suppression(score, choose_maxima=True):
    score_ptr = score.ctypes.data_as(POINTER(c_float))
    w = score.shape[1]
    h = score.shape[0]

    result = np.zeros(score.shape, dtype=np.uint8)
    result_ptr = result.ctypes.data_as(POINTER(c_ubyte))
    choose_maxima = 1 if choose_maxima else -1

    ext_dll.NonExtremaSuppression(
        score_ptr, c_int(w), c_int(h), result_ptr, c_int(choose_maxima)
    )

    yx = result.nonzero()

    return yx[1], yx[0]


def non_extrema_suppression_multi_scale(score_pyramid, choose_maxima=True):
    score_ptr = score_pyramid.ctypes.data_as(POINTER(c_float))
    c = score_pyramid.shape[0]
    w = score_pyramid.shape[2]
    h = score_pyramid.shape[1]

    result = np.zeros(score_pyramid.shape, dtype=np.uint8)
    result_ptr = result.ctypes.data_as(POINTER(c_ubyte))
    choose_maxima = 1 if choose_maxima else -1

    ext_dll.NonExtremaSuppressionMultiScale(
        score_ptr, c_int(c), c_int(w), c_int(h), result_ptr, c_int(choose_maxima)
    )

    syx = result.nonzero()

    return syx[0], syx[2], syx[1]


def compute_subpix_quadratic(score, x_, y_):
    # print('computing subpix')
    score_ptr = score.ctypes.data_as(POINTER(c_float))
    w = score.shape[1]
    h = score.shape[0]

    x = x_.copy().astype(np.float32)
    y = y_.copy().astype(np.float32)
    x_ptr = x.ctypes.data_as(POINTER(c_float))
    y_ptr = y.ctypes.data_as(POINTER(c_float))

    ext_dll.CalcSubpixel(score_ptr, c_int(w), c_int(h), c_int(x.size), x_ptr, y_ptr)

    good = np.logical_and(
        np.logical_and(x >= 0, x <= w - 1), np.logical_and(y >= 0, y <= h - 1)
    )

    x = x[good]
    y = y[good]

    return x, y


def distance_based_matching(kx1_, ky1_, kx2_, ky2_, tform_):
    kx1 = kx1_.copy()
    ky1 = ky1_.copy()
    kx2 = kx2_.copy()
    ky2 = ky2_.copy()
    tform = tform_.copy()

    if (
        kx1.dtype != np.float32
        or ky1.dtype != np.float32
        or kx2.dtype != np.float32
        or ky2.dtype != np.float32
        or tform.dtype != np.float32
    ):

        print("distance_based_matching: data type is not float32")
        print("kx1.dtype = {}".format(kx1.dtype))
        print("ky1.dtype = {}".format(ky1.dtype))
        print("kx2.dtype = {}".format(kx2.dtype))
        print("ky2.dtype = {}".format(ky2.dtype))
        print("tform.dtype = {}".format(tform.dtype))

    kx1_ptr = kx1.ctypes.data_as(POINTER(c_float))
    ky1_ptr = ky1.ctypes.data_as(POINTER(c_float))
    kx2_ptr = kx2.ctypes.data_as(POINTER(c_float))
    ky2_ptr = ky2.ctypes.data_as(POINTER(c_float))
    tform_ptr = tform.ctypes.data_as(POINTER(c_float))

    nkeys1 = c_int(kx1.size)
    nkeys2 = c_int(kx2.size)

    nn_idx = np.zeros(kx1.size, np.int32)
    nn_dist = np.zeros(kx1.size, np.float32)

    nn_idx_ptr = nn_idx.ctypes.data_as(POINTER(c_int))
    nn_dist_ptr = nn_dist.ctypes.data_as(POINTER(c_float))

    ext_dll.DistanceBasedMatching(
        kx1_ptr,
        ky1_ptr,
        nkeys1,
        kx2_ptr,
        ky2_ptr,
        nkeys2,
        tform_ptr,
        nn_idx_ptr,
        nn_dist_ptr,
    )

    return nn_idx, nn_dist
