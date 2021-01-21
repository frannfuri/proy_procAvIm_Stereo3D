# imports
from utils.calib import obtain_points, calib_cam, get_fl_exif, calc_proj_error
import numpy as np

#Camera1 --> Rigth side
#Camera2 --> Left side

if __name__ == '__main__':
    # ============================================
    # Camera calibration
    # ============================================
    imgs_path = './calibration_imgs/*'
    chess_size = (9, 6) # (7,5)
    num_iter_crit = 30
    accuracy_crit = 0.001
    win_size_ref = 11
    obj_points, img_points, imgs_with_pattern = obtain_points(chess_size,     #obj_points --> dim (54,3);  img_points --> dim (54, 1, 2)
                                           num_iter_crit,
                                           accuracy_crit,
                                           win_size_ref,
                                           imgs_path)


    # Guardar obj_points e img_points
    np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/obj_points0", obj_points)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/img_points0", img_points)


    # Calibrar camara y guardar parametros
    pixels_calib_imgs = (2016, 1512)
    ret, K, dist, rvecs, tvecs = calib_cam(obj_points, img_points, pixels_calib_imgs, '0')
    # dims
    # ret --> 1,  K --> (3,3),  dist --> (5),  rvecs --> (7,3),  tvecs --> (7,3)
    # Obtención Focal length por Exif data
    # foc len tupla  , foc len decimal
    #focal_length_exif, focal_length = get_fl_exif(imgs_path)
    print('RMS1: ' + str(ret))


    # Error de proyeccion
    print('Projection error:')
    print('Cam: ')
    total_error = calc_proj_error(obj_points, img_points, rvecs, tvecs, K, dist)