import cv2
import numpy as np

# Evaluacion visual de la Calibracion

def undistort_way(img, mtx, dist, newcameramtx, w, h):
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult_undistort.jpg', dst)

def remapping_way(img, mtx, dist, newcameramtx, w, h):
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult_remapping.png', dst)

if __name__ == '__main__':
    # Config
    img_path = "/home/franrosi/PycharmProjects/StereoRecons/test_imgs/20210119_013013.jpg"
    mtx = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K2.npy")
    dist = np.load("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist2.npy")

    # Preliminaries
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                      (w,h),1,(w,h))
    # Undistort
    #undistort_way(img, mtx, dist, newcameramtx, w, h)
    remapping_way(img, mtx, dist, newcameramtx, w, h)
