from itertools import cycle

import open3d as o3d
import numpy as np


class Open3dVisualizer:
    def __init__(self, window_name="visualization", window_width=800, window_height=600, color_depth=True,
                 max_depth=100.0):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name=window_name, width=window_width, height=window_height)

        render_options = self.visualizer.get_render_option()
        render_options.background_color = (0.498, 0.788, 1)
        render_options.light_on = True

        self.pcd = None
        self.is_color_depth = color_depth
        self.max_depth_color_value = max_depth
        self.default_colors = [(1.0, 1.0, 1.0),
                               (0.0, 0.0, 1.0),
                               (0.0, 1.0, 0.0),
                               (1.0, 0.0, 0.0),
                               (1.0, 1.0, 0.0),
                               (1.0, 0.0, 1.0),
                               (0.0, 1.0, 1.0),
                               (0.5, 0.5, 1.0),
                               (0.5, 1.0, 0.5),
                               (1.0, 0.5, 0.5)]

    def wait_for_window_closure(self):
        self.visualizer.run()

    def destroy_window(self):
        self.visualizer.destroy_window()

    def __get_points_color_scale(self, points, color, axis=2):
        column = 1 - np.expand_dims(points[:, axis], axis=1) / self.max_depth_color_value
        return np.concatenate((column * color[0], column * color[1], column * color[2]), axis=1)

    def __colour_points(self, pcd, color):
        if self.is_color_depth:
            pcd.colors = o3d.utility.Vector3dVector(self.__get_points_color_scale(np.asarray(pcd.points), color=color))
        else:
            pcd.paint_uniform_color(color)

    def add_points(self, points):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.__colour_points(self.pcd, (1.0, 1.0, 1.0))
        self.visualizer.add_geometry(self.pcd)
        self.visualizer.poll_events()
        # visualizer.update_renderer()

    def change_points(self, new_points):
        if self.pcd is None:
            self.add_points(new_points)
        else:
            self.pcd.points = o3d.utility.Vector3dVector(new_points)
            self.__colour_points(self.pcd, (1.0, 1.0, 1.0))
            self.visualizer.update_geometry(self.pcd)
            self.visualizer.poll_events()
            # visualizer.update_renderer()

    def show_clouds(self, point_clouds):
        self.visualizer.clear_geometries()
        for point_cloud, color in zip(point_clouds, cycle(self.default_colors)):
            self.__add_cloud(point_cloud, color)
        self.visualizer.poll_events()

    def __add_cloud(self, point_cloud, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        self.__colour_points(pcd, color)
        self.visualizer.add_geometry(pcd)
