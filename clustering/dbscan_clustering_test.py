import numpy as np
from cloud_generation import generate_points_with_pix_coordinates

from dbscan_clustering import create_clusters
from visualizer.open3d_visualizer import Open3dVisualizer


if __name__ == '__main__':
    filename = "C:\\Users\\Michal\\Pictures\\Test\\test6_depth.npy"
    depth_data = original_data = np.load(filename).squeeze()
    print(depth_data.shape)
    points = generate_points_with_pix_coordinates(depth_data)
    # X = modify_points(X, width_modifier=1, height_modifier=1, depth_modifier=20)
    clusters = create_clusters(points, eps=2, min_samples=5)
    visualizer = Open3dVisualizer(max_depth=300)
    visualizer.show_clouds(clusters)
    visualizer.wait_for_window_closure()
