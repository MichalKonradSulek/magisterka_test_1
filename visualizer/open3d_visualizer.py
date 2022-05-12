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
        self.was_already_geometry_rendered = False
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
        self.destroy_window()

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

    @staticmethod
    def __calculate_center_of_cloud(point_clouds):
        maximum = []
        minimum = []
        for cloud in point_clouds:
            maximum.append(cloud.max(axis=0))
            minimum.append(cloud.min(axis=0))
        maximum = np.array(maximum)
        minimum = np.array(minimum)
        max_val = maximum.max(axis=0)
        min_val = minimum.min(axis=0)
        center = (max_val + min_val) / 2
        return center

    def __setup_camera_view(self, point_clouds):
        self.visualizer.get_view_control().set_front((0.5, -0.2, -1.0))
        self.visualizer.get_view_control().set_up((0.0, -1.0, 0.0))
        look_at_point = Open3dVisualizer.__calculate_center_of_cloud(point_clouds)
        look_at_point[2] = 0.0
        print(look_at_point)
        self.visualizer.get_view_control().set_lookat(look_at_point)
        self.visualizer.get_view_control().set_zoom(0.5)

    def show_clouds(self, point_clouds):
        self.visualizer.clear_geometries()
        for point_cloud, color in zip(point_clouds, cycle(self.default_colors)):
            self.__add_cloud(point_cloud, color, not self.was_already_geometry_rendered)
        if not self.was_already_geometry_rendered:
            self.__setup_camera_view(point_clouds)
            self.was_already_geometry_rendered = True
        self.visualizer.poll_events()

    def __add_cloud(self, point_cloud, color, reset_bb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        self.__colour_points(pcd, color)
        self.visualizer.add_geometry(pcd, reset_bounding_box=reset_bb)
