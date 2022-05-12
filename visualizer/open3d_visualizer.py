import itertools

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

    def wait_for_window_closure(self):
        self.visualizer.run()

    def destroy_window(self):
        self.visualizer.destroy_window()

    def __get_points_grey_scale(self, points, axis=2):
        column = 1 - np.expand_dims(points[:, axis], axis=1) / self.max_depth_color_value
        return np.concatenate((column, column, column), axis=1)

    def add_points(self, points):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        if self.is_color_depth:
            self.pcd.colors = o3d.utility.Vector3dVector(self.__get_points_grey_scale(points))
        else:
            self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.visualizer.add_geometry(self.pcd)
        self.visualizer.poll_events()
        # visualizer.update_renderer()

    def change_points(self, new_points):
        if self.pcd is None:
            self.add_points(new_points)
        else:
            self.pcd.points = o3d.utility.Vector3dVector(new_points)
            if self.is_color_depth:
                self.pcd.colors = o3d.utility.Vector3dVector(self.__get_points_grey_scale(new_points))
            self.visualizer.update_geometry(self.pcd)
            self.visualizer.poll_events()
            # visualizer.update_renderer()


def get_data_from_file(filepath):
    print("file: " + filepath)
    original_data = np.load(filepath)
    return original_data[0, 0, :, :]


def get_points_coordinates(array, step_width: int = 1, step_height: int = 1):
    x = []
    y = []
    z = []
    width = array.shape[1]
    height = array.shape[0]
    for v, h in itertools.product(range(0, height, step_height), range(0, width, step_width)):
        x.append(h)
        z.append(height - v - 1)
        y.append(array[v, h])
    return x, y, z


def run_window(window):
    window.run()


