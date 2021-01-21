import cv2
import numpy as np
import matplotlib.pyplot as plt

#Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

if __name__ == '__main__':
    # Load camera parameters
    ret = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/ret.npy')
    K1 = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/K1.npy')
    dist1 = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/D1.npy')
    K2 = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/K2.npy')
    dist2 = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/D2.npy')
    R = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/R.npy')
    T = np.load('/home/franrosi/PycharmProjects/StereoRecons/stereo_params/T.npy')

    # Specify image paths
    # TRAIN IMAGES
    ##img_path1 = '/home/franrosi/PycharmProjects/StereoRecons/train_final/img_test11_1.jpg'
    #img_path2 = '/home/franrosi/PycharmProjects/StereoRecons/train_final/img_test11_2.jpg'

    # TEST IMAGES
    img_path1 = '/home/franrosi/PycharmProjects/StereoRecons/test_final/img11_1.jpg'
    img_path2 = '/home/franrosi/PycharmProjects/StereoRecons/test_final/img11_2.jpg'
    # Load pictures
    img_1 = cv2.imread(img_path1)
    img_2 = cv2.imread(img_path2)
    # Get height and width. Both pictures have to be same size
    h, w = img_2.shape[:2]

    # Get optimal camera matrix for better undistortion
    new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(K1, dist1, (w, h), 1, (w, h))  # Undistort images
    new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(K2, dist2, (w, h), 1, (w, h))
    img_1_undistorted = cv2.undistort(img_1, K1, dist1, None, new_camera_matrix1)
    img_2_undistorted = cv2.undistort(img_2, K2, dist2, None, new_camera_matrix2)


    pixels_imgs = (2016, 1512)  # image size
    flags = 0

    # R: rect_trans, P: proj_mats, Q: disp_to_depth_mat
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, dist1,
                                                                      K2, dist2,
                                                                      pixels_imgs,
                                                                      R, T, flags=flags, alpha=0)

    undistortion_map1x, undistortion_map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, pixels_imgs,
                                                    cv2.CV_32FC1)
    undistortion_map2x, undistortion_map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, pixels_imgs,
                                                    cv2.CV_32FC1)

    dst1 = cv2.remap(img_1_undistorted, undistortion_map1x, undistortion_map1y, cv2.INTER_NEAREST)
    dst2 = cv2.remap(img_2_undistorted, undistortion_map2x, undistortion_map2y, cv2.INTER_NEAREST)
    # Se reduce la dimension de las imagenes para aumentar velocidad de procesamiento
    # y para poder ajustar los parametros del calculo del mapa de disparidad
    # en el codigo tunning_matching.py
    dst1_downsampled = downsample_image(dst1, 3)
    dst2_downsampled = downsample_image(dst2, 3)

    h, w = dst1_downsampled.shape[:2]
    # Empiricamente la matriz Q que entrega stereoRectify no fue muy buena
    # Por lo anterior se realiza una estimacion basandose en uno de los ejemplos del github de OpenCV
    # Estimation of Q (different from obtained by stereoRectify)
    focal_length = 0.8 * w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0, 0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0, -focal_length], # so that y-axis looks up
                    [0, 0, 1, 0]])


    # Undistorted and remap images
    cv2.imwrite('dst1_remap.jpg', dst1_downsampled)
    cv2.imwrite('dst2_remap.jpg', dst2_downsampled)

############################################

    # Los parametros se tunearon en el codigo implementado
    # en tunning_matching.py, donde se genera una interfaz grafica
    # para modificar los parametros en tiempo real y ver el resultado

    # Parametros elegidos
    block_size = 9  # 19
    min_disp = 2  # 10
    num_disp = 46
    disp12MaxDiff = 4
    speckleRange = 5
    speckleWindowSize = 49
    P1 = 1789  # 600
    P2 = 867
    preFilterCap = 59
    uniquenessRatio = 15
    smallerBlockSize = 1

    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block_size,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange,
                                   disp12MaxDiff=disp12MaxDiff,
                                   P1= P1,
                                   P2= P2,
                                   preFilterCap=preFilterCap)

    # Compute disparity map
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(dst1_downsampled, dst2_downsampled).astype(np.float32)/16.0

    # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
    plt.figure()
    plt.imshow(dst1_downsampled)
    plt.figure()
    plt.imshow(dst2_downsampled)
    plt.figure()
    plt.imshow(disparity_map, 'gray')
    plt.show()

    np.save('Q.npy', Q)
    np.save('disparity_map.npy', disparity_map)