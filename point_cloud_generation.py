import math

import numpy as np
from visualizer.open3d_visualizer import Open3dVisualizer


def generate_3d_point_cloud(depth_array, f=0.00405, pix_size=0.00000112):
    # To się da przyspieszyć, tworząc gotową macierz o wymiarze zdjęcia i mnożąc jedną przez drugą
    width_pix = depth_array.shape[1]
    height_pix = depth_array.shape[0]
    cpw = (width_pix - 1) / 2.0
    cph = (height_pix - 1) / 2.0
    result = []
    for ih, iw in np.ndindex(depth_array.shape):
        depth = depth_array[ih, iw]
        mw = (iw - cpw) * pix_size
        mh = (ih - cph) * pix_size
        md = math.sqrt(mw ** 2 + mh ** 2 + f ** 2)
        k = depth / md
        result.append([mw * k, mh * k, f * k])
    return np.array(result)


def generate_3d_point_cloud_article(depth_array, f=0.00405, pix_size=0.00000112):
    # To się da przyspieszyć, tworząc gotową macierz o wymiarze zdjęcia i mnożąc jedną przez drugą
    width_pix = depth_array.shape[1]
    height_pix = depth_array.shape[0]
    result = []
    for ih, iw in np.ndindex(depth_array.shape):
        depth = depth_array[ih, iw]
        x = (depth + f) * (width_pix / 2.0 - iw - 1) * pix_size / f
        y = (depth + f) * (height_pix / 2.0 - ih - 1) * pix_size / f
        result.append([-x, -y, depth])
    return np.array(result)


def generate_points_with_pix_coordinates(depth_array):
    result = []
    for ih, iw in np.ndindex(depth_array.shape):
        result.append([iw, ih, depth_array[ih, iw]])
    return np.array(result)


if __name__ == '__main__':
    filepath = "C:\\Users\\Michal\\Pictures\\Test\\test5_depth.npy"
    print("file: " + filepath)
    depth_array = np.load(filepath)
    point_cloud = generate_3d_point_cloud(depth_array.squeeze(), f=0.00405, pix_size=0.0000112)
    point_cloud_article = generate_3d_point_cloud_article(depth_array.squeeze(), f=0.00405, pix_size=0.0000112)
    # point_cloud = generate_points_with_pix_coordinates(depth_array.squeeze())
    visualizer = Open3dVisualizer()
    # visualizer.add_points(point_cloud)
    visualizer.show_clouds([point_cloud, point_cloud_article])
    visualizer.wait_for_window_closure()
