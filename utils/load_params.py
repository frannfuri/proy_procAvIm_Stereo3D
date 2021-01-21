import numpy as np

def load_cam_params(cam_num):
    if cam_num == '1':
        ret = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/ret1.npy")
        K = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K1.npy")
        dist = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist1.npy")
        rvecs = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/rvecs1.npy")
        tvecs = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/tvecs1.npy")
    else:
        ret = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/ret2.npy")
        K = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K2.npy")
        dist = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist2.npy")
        rvecs = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/rvecs2.npy")
        tvecs = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/tvecs2.npy")
    return ret, K, dist, rvecs, tvecs