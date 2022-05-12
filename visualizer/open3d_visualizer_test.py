import time
import numpy as np

from point_cloud_generation import generate_points_with_pix_coordinates
from open3d_visualizer import Open3dVisualizer


if __name__ == "__main__":
    filename = "C:\\Users\\Michal\\Pictures\\Test\\test6_depth.npy"
    original_data = np.load(filename)
    depth = original_data.squeeze()
    print(depth.shape)
    xyz = generate_points_with_pix_coordinates(depth)
    print(xyz)

    split_points = np.split(xyz, 20)

    visualizer = Open3dVisualizer(max_depth=100)
    visualizer.show_clouds(split_points)
    for i in range(10):
        for j, v in enumerate(split_points):
            split_points[j] = split_points[j] + 10
        visualizer.show_clouds(split_points)
        time.sleep(1)
    visualizer.wait_for_window_closure()
