import numpy as np
from utils.load_params import load_cam_params
from utils.calib import calib_stereo
import cv2

pixels_calib_imgs = (2016, 1512) # image size

# Load obj and img points
img_points1 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/img_points1.npy")
img_points2 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/img_points2.npy")
obj_points = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/obj_points.npy")

# Load camera parameters
ret1, K1, dist1, rvecs1, tvecs1 = load_cam_params('1')
ret2, K2, dist2, rvecs2, tvecs2 = load_cam_params('2')

K1 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K1.npy")
dist1 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist1.npy")
K2 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K2.npy")
dist2 = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist2.npy")

# config
flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retval, K1, D1, K2, D2, R, T = calib_stereo(obj_points, img_points1, img_points2, pixels_calib_imgs, K1, K2, dist1, dist2, flags, stereocalib_criteria)
print('RMS: ' + str(retval))