import numpy as np
from visualizer.open3d_visualizer import Open3dVisualizer


def generate_3d_point_cloud(depth_array, f=1.7):
    # To się da przyspieszyć, tworząc gotową macierz o wymiarze zdjęcia i mnożąc jedną przez drugą
    width_pix = depth_array.shape[1]
    height_pix = depth_array.shape[0]
    cpw = (width_pix - 1) / 2.0
    cph = (height_pix - 1) / 2.0
    result = []
    for ih, iw in np.ndindex(depth_array.shape):
        depth = depth_array[ih, iw]
        height = (ih - cph) * depth / f
        width = (iw - cpw) * depth / f
        result.append([width, height, depth])
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
    # point_cloud = generate_3d_point_cloud(depth_array.squeeze())
    point_cloud = generate_points_with_pix_coordinates(depth_array.squeeze())
    visualizer = Open3dVisualizer()
    visualizer.add_points(point_cloud)
    visualizer.wait_for_window_closure()
