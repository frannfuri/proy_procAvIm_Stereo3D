import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

#Function to create point cloud file
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    dst1_remap = cv2.imread('dst1_remap.jpg')
    dst2_remap = cv2.imread('dst2_remap.jpg')
    print('\nGenerating the 3D map...')
    Q = np.load('Q.npy')
    disparity_map = np.load('disparity_map.npy')
    points_3D = cv2.reprojectImageTo3D(disparity=disparity_map, Q=Q)
    
    # Get color points
    colors = cv2.cvtColor(dst1_remap, cv2.COLOR_BGR2RGB)

    # Get rid of points with value 0 (i.e. sin profundidad)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    # Output file
    output_file = 'reconstructed_3D.ply'

    # Generate point cloud
    print('\n Creating the output file... \n')
    write_ply(output_file, output_points, output_colors)
