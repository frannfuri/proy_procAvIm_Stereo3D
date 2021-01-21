import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Slider

dst1_downsampled = cv2.imread('dst1_remap.jpg')
dst2_downsampled = cv2.imread('dst2_remap.jpg')



if __name__ == '__main__':
    plt.imshow(dst1_downsampled)
    plt.figure()
    plt.imshow(dst2_downsampled)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.75)
    # Create axes for sliders
    ax_block = fig.add_axes([0.1, 0.98, 0.8, 0.01])
    ax_block.spines['top'].set_visible(True)
    ax_block.spines['right'].set_visible(True)
    ax_minDisp = fig.add_axes([0.1, 0.95, 0.8, 0.01])
    ax_minDisp.spines['top'].set_visible(True)
    ax_minDisp.spines['right'].set_visible(True)
    ax_numDisp = fig.add_axes([0.1, 0.92, 0.8, 0.01])
    ax_numDisp.spines['top'].set_visible(True)
    ax_numDisp.spines['right'].set_visible(True)
    ax_disp12 = fig.add_axes([0.1, 0.89, 0.8, 0.01])
    ax_disp12.spines['top'].set_visible(True)
    ax_disp12.spines['right'].set_visible(True)
    ax_speckR = fig.add_axes([0.1, 0.86, 0.8, 0.01])
    ax_speckR.spines['top'].set_visible(True)
    ax_speckR.spines['right'].set_visible(True)
    ax_speckW = fig.add_axes([0.1, 0.83, 0.8, 0.01])
    ax_speckW.spines['top'].set_visible(True)
    ax_speckW.spines['right'].set_visible(True)
    ax_P10 = fig.add_axes([0.1, 0.8, 0.8, 0.01])
    ax_P10.spines['top'].set_visible(True)
    ax_P10.spines['right'].set_visible(True)
    ax_P20 = fig.add_axes([0.1, 0.77, 0.8, 0.01])
    ax_P20.spines['top'].set_visible(True)
    ax_P20.spines['right'].set_visible(True)
    ax_preFilter = fig.add_axes([0.1, 0.74, 0.8, 0.01])
    ax_preFilter.spines['top'].set_visible(True)
    ax_preFilter.spines['right'].set_visible(True)
    ax_uniqRatio = fig.add_axes([0.1, 0.71, 0.8, 0.01])
    ax_uniqRatio.spines['top'].set_visible(True)
    ax_uniqRatio.spines['right'].set_visible(True)

    # Plot default parameters
    block_size0 = 8 #19
    min_disp0 = 7 #10
    num_disp0 = 71
    disp12MaxDiff0 = 9
    speckleRange0 = 5
    speckleWindowSize0 = 9
    P10 = 1792 #600
    P20 = 865
    preFilterCap0 = 25
    uniquenessRatio0 = 15
    smallerBlockSize0 = 1

    # Create sliders
    s_block = Slider(valinit=block_size0, ax=ax_block, label='block size ', valmin=0, valmax=255,valfmt = ' %i')
    s_minDisp = Slider(valinit=min_disp0, ax=ax_minDisp, label='min disp ', valmin=0, valmax=255, valfmt=' %i')
    s_numDisp = Slider(valinit=num_disp0, ax=ax_numDisp, label='num disp ', valmin=16, valmax=2048, valfmt=' %i')
    s_disp12 = Slider(valinit=disp12MaxDiff0, ax=ax_disp12, label='disp12 ', valmin=-1, valmax=255, valfmt=' %i')
    s_speckR = Slider(valinit=speckleRange0, ax=ax_speckR, label='speckR ', valmin=-1, valmax=255, valfmt=' %i')
    s_speckW = Slider(valinit=speckleWindowSize0, ax=ax_speckW, label='speckW ', valmin=0, valmax=255, valfmt=' %i')
    s_P10 = Slider(valinit=P10, ax=ax_P10, label='P10 ', valmin=0, valmax=2048, valfmt=' %i')
    s_P20 = Slider(valinit=P20, ax=ax_P20, label='P20 ', valmin=0, valmax=2048, valfmt=' %i')
    s_preFilter = Slider(valinit=preFilterCap0, ax=ax_preFilter, label='preFilter ', valmin=1, valmax=63, valfmt=' %i')
    s_uniqRatio = Slider(valinit=uniquenessRatio0, ax=ax_uniqRatio, label='uniqRatio ', valmin=0, valmax=255, valfmt=' %i')




    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp0,
                                   numDisparities=num_disp0,
                                   blockSize=block_size0,
                                   uniquenessRatio=uniquenessRatio0,
                                   speckleWindowSize=speckleWindowSize0,
                                   speckleRange=speckleRange0,
                                   disp12MaxDiff=disp12MaxDiff0,
                                   P1=P10,  # 8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=P20,  # 32 * 3 * win_size ** 2)  # 32*3*win_size**2)
                                   preFilterCap=preFilterCap0)

    # Compute disparity map
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(dst1_downsampled, dst2_downsampled)

    # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
    ax.imshow(disparity_map, 'gray')


    def update(val):
        block_size0 = int(s_block.val)
        min_disp0 = int(s_minDisp.val)
        num_disp0 = int(s_numDisp.val)
        disp12MaxDiff0 = int(s_disp12.val)
        #speckleRange0 = int(s_speckR.val)
        speckleWindowSize0 = int(s_speckW.val)
        P10 = int(s_P10.val)
        P20 = int(s_P20.val)
        preFilterCap0 = int(s_preFilter.val)
        #uniquenessRatio0 = int(s_uniqRatio.val)
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp0,
                                       numDisparities=num_disp0,
                                       blockSize=block_size0,
                                       uniquenessRatio=uniquenessRatio0,
                                       speckleWindowSize=speckleWindowSize0,
                                       speckleRange=speckleRange0,
                                       disp12MaxDiff=disp12MaxDiff0,
                                       P1=P10,  # 8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                       P2=P20,  # 32 * 3 * win_size ** 2)  # 32*3*win_size**2)
                                       preFilterCap=preFilterCap0)

        # Compute disparity map
        print("\nComputing the disparity  map...")
        disparity_map = stereo.compute(dst1_downsampled, dst2_downsampled)
        ax.imshow(disparity_map, 'gray')
    s_block.on_changed(update)
    s_minDisp.on_changed(update)
    s_numDisp.on_changed(update)
    s_disp12.on_changed(update)
    #s_speckR.on_changed(update)
    s_speckW.on_changed(update)
    s_P10.on_changed(update)
    s_P20.on_changed(update)
    s_preFilter.on_changed(update)
    #s_uniqRatio.on_changed(update)

    plt.show()

