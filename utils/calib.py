import cv2
import numpy as np
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

def obtain_points(chess_size, num_iter_crit, accuracy_crit, win_size_ref,
                  imgs_path):
    # Define size of chessboard target.
    # Define arrays to save detected points
    obj_points = [] #3D points in real world space
    img_points = [] #3D points in image plane
    # Prepare grid and points to display
    objp = np.zeros((np.prod(chess_size),3),dtype=np.float32)

    # Todas las combinaciones posibles
    objp[:,:2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1,2)

    # read images
    calibration_paths = glob.glob(imgs_path)
    # Iterate over images to find intrinsic matrix
    true_rets = 0
    imgs_with_pattern = []
    for image_path in tqdm(calibration_paths):
    # Load image
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Image loaded, Analizying...")
        # find chessboard corners
        ret,corners = cv2.findChessboardCorners(gray_image, chess_size, None)
        # si se obtiene el patron --> ret = True
        # corner points --> corners

        # Si se encontraron, agregar object points, image points (despues de refinarlos)
        if ret == True:
            true_rets += 1
            print("Chessboard detected!")
            print(image_path)
            # define criteria for subpixel accuracy
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_iter_crit, accuracy_crit)
            #refine corner location (to subpixel accuracy) based on criteria.
            corners2 = cv2.cornerSubPix(gray_image, corners, (int(win_size_ref/2),int(win_size_ref/2)), (-1,-1), criteria)
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(image, chess_size, corners2, ret)
            cv2.imshow('img',image)
            cv2.waitKey(500)
            imgs_with_pattern.append(image_path)
    cv2.destroyAllWindows()
    print('\nImages with pattern identified: ' + str(imgs_with_pattern))
    print('\nNumber of True pattern returns: ' + str(true_rets))
    return obj_points, img_points, imgs_with_pattern

def calib_cam(obj_points, img_points, pixels_calib_imgs,cam_num):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, pixels_calib_imgs, None, None)
    # save parameters into numpy file
    if cam_num == '1':
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/ret1", ret)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K1", K)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist1", dist)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/rvecs1", rvecs)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/tvecs1", tvecs)
    if cam_num == '0':
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/ret0", ret)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K0", K)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist0", dist)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/rvecs0", rvecs)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/tvecs0", tvecs)
    else:
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/ret2", ret)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/K2", K)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/dist2", dist)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/rvecs2", rvecs)
        np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/tvecs2", tvecs)
    return ret, K, dist, rvecs, tvecs

def get_fl_exif(imgs_path):
    # Obtener exif data para obtener Focal length
    exif_img = PIL.Image.open(glob.glob(imgs_path)[0])
    exif_data = {PIL.ExifTags.TAGS[k]: v
                 for k, v in exif_img._getexif().items()
                 if k in PIL.ExifTags.TAGS}
    # Focal length como tupla
    focal_length_exif = exif_data['FocalLength']
    # Focal length como decimal
    focal_length = focal_length_exif[0] / focal_length_exif[1]
    # Save focal length
    np.save("./camera_params/FocalLength", focal_length)
    return focal_length_exif, focal_length

# Reprojection error
def calc_proj_error(obj_points, img_points, rvecs, tvecs, K, dist):
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    total_error = mean_error / len(obj_points)
    print('Total error: ' + str(total_error))
    return total_error


def calib_stereo(obj_points, img_points1, img_points2, pixels_calib_imgs, K1, K2, dist1, dist2,flags,criteria):
    retval, K1, D1, K2, D2, R, T,_,_ = cv2.stereoCalibrate(objectPoints=obj_points, imagePoints1=img_points1, imagePoints2=img_points2,
                        cameraMatrix1=K1, distCoeffs1=dist1, cameraMatrix2=K2, distCoeffs2=dist2,
                        imageSize=pixels_calib_imgs, flags=flags, criteria = criteria)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/ret", retval)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/K1", K1)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/D1", D1)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/K2", K2)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/D2", D2)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/R", R)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/stereo_params/T", T)
    return retval, K1, D1, K2, D2, R, T

