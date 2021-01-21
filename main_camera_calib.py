# imports
from utils.calib import obtain_points, calib_cam, get_fl_exif, calc_proj_error
import numpy as np

#Camera1 --> Rigth side
#Camera2 --> Left side

if __name__ == '__main__':
    # ============================================
    # Camera calibration
    # ============================================
    imgs1_path = './calibration_imgs1/*'
    imgs2_path = './calibration_imgs2/*'
    chess_size = (9, 6) # (7,5)
    num_iter_crit = 30
    accuracy_crit = 0.001
    win_size_ref = 11
    obj_points1, img_points1, imgs_with_pattern1 = obtain_points(chess_size,     #obj_points --> dim (54,3);  img_points --> dim (54, 1, 2)
                                           num_iter_crit,
                                           accuracy_crit,
                                           win_size_ref,
                                           imgs1_path)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/imgs_with_pattern1", imgs_with_pattern1)
    obj_points2, img_points2, imgs_with_pattern2 = obtain_points(chess_size,     #obj_points --> dim (54,3);  img_points --> dim (54, 1, 2)
                                           num_iter_crit,
                                           accuracy_crit,
                                           win_size_ref,
                                           imgs2_path)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/imgs_with_pattern2", imgs_with_pattern2)
    # REORDER: To match img_points1 with img_points2
    #order1 = [0,1,2,4,5,6,7,9,10,11,3,8]
    #order2 = [9,5,10,12,0,8,2,1,11,3,4,6,7]
    order1 = [0,1,2,3,4,5,7,10,11,12,6,8,9]
    order2 = [8,5,9,11,0,7,2,1,10,3,4,6]
    img_points1 = [img_points1[i] for i in order1]
    img_points2 = [img_points2[i] for i in order2]
    img_points1 = img_points1[0:10]
    img_points2 = img_points2[0:10]

    #obj_points1 = obj_points1[0:11]
    obj_points1 = obj_points1[0:10]

    # Guardar obj_points e img_points
    np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/obj_points", obj_points1)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/img_points1", img_points1)
    np.save("/home/franrosi/PycharmProjects/StereoRecons/camera_params/img_points2", img_points2)


    # Calibrar camara y guardar parametros
    pixels_calib_imgs = (2016, 1512)
    ret1, K1, dist1, rvecs1, tvecs1 = calib_cam(obj_points1, img_points1, pixels_calib_imgs, '1')
    ret2, K2, dist2, rvecs2, tvecs2 = calib_cam(obj_points1, img_points2, pixels_calib_imgs, '2')
    # dims
    # ret --> 1,  K --> (3,3),  dist --> (5),  rvecs --> (7,3),  tvecs --> (7,3)
    # Obtención Focal length por Exif data
    # foc len tupla  , foc len decimal
    #focal_length_exif, focal_length = get_fl_exif(imgs_path)
    print('RMS1: ' + str(ret1))
    print('RMS2: ' + str(ret2))


    # Error de proyeccion
    print('Projection error:')
    print('Cam1: ')
    total_error1 = calc_proj_error(obj_points1, img_points1, rvecs1, tvecs1, K1, dist1)
    print('Cam2: ')
    total_error2 = calc_proj_error(obj_points1, img_points2, rvecs2, tvecs2, K2, dist2)
