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

    visualizer = Open3dVisualizer(max_depth=1000)
    visualizer.add_points(xyz)
    for i in range(10):
        xyz = xyz + 10
        visualizer.change_points(xyz)
        time.sleep(1)
    visualizer.wait_for_window_closure()
