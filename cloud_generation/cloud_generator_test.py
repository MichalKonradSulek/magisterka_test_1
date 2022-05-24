import numpy as np

from visualizer.open3d_visualizer import Open3dVisualizer
from cloud_generation.point_cloud_generator import PointCloudGenerator

if __name__ == '__main__':
    filepath = "C:\\Users\\Michal\\Pictures\\Test\\test5_depth.npy"
    print("file: " + filepath)
    depth_array = np.load(filepath)
    point_cloud_generator = PointCloudGenerator(640, 192, 0.0043008, 0.0024192, 0.00405)
    # point_cloud = generate_3d_point_cloud(depth_array.squeeze(), f=0.00405, pix_size=0.0000112)
    point_cloud = point_cloud_generator.generate(depth_array.squeeze())
    # point_cloud_article = generate_3d_point_cloud_article(depth_array.squeeze(), f=0.00405, pix_size=0.0000112)
    # point_cloud = generate_points_with_pix_coordinates(depth_array.squeeze())
    visualizer = Open3dVisualizer()
    visualizer.add_points(point_cloud)
    # visualizer.show_clouds([point_cloud, point_cloud_article])
    visualizer.wait_for_window_closure()
