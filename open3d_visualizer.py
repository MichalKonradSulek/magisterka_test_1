import itertools
import math
import time

import open3d as o3d
import numpy as np
import threading


class Open3dVisualizer:
    def __init__(self, window_name="visualization", window_width=800, window_height=600):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name=window_name, width=window_width, height=window_height)

        render_options = self.visualizer.get_render_option()
        render_options.background_color = (0.8, 0.8, 1.0)
        render_options.light_on = True

        self.pcd = None

    def wait_for_window_closure(self):
        self.visualizer.run()

    def add_points(self, points):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.visualizer.add_geometry(self.pcd)
        self.visualizer.poll_events()
        # visualizer.update_renderer()

    def change_points(self, new_points):
        self.pcd.points = o3d.utility.Vector3dVector(new_points)
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


if __name__ == "__main__":
    # # Generate some n x 3 matrix using a variant of sync function.
    # x = np.linspace(-3, 3, 201)
    # mesh_x, mesh_y = np.meshgrid(x, x)
    # z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    # z_norm = (z - z.min()) / (z.max() - z.min())
    # xyz = np.zeros((np.size(mesh_x), 3))
    # xyz[:, 0] = np.reshape(mesh_x, -1)
    # xyz[:, 1] = np.reshape(mesh_y, -1)
    # xyz[:, 2] = np.reshape(z_norm, -1)
    # print("Printing numpy array used to make Open3D pointcloud ...")
    # print(xyz)

    filename = "C:\\Users\\Michal\\Pictures\\Test\\test6_depth.npy"
    depth = get_data_from_file(filename)
    x, y, z = get_points_coordinates(depth)
    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    # o3d.visualization.draw([pcd])

    visualizer = Open3dVisualizer()
    visualizer.add_points(xyz)
    for i in range(10):
        xyz = xyz + 10
        visualizer.change_points(xyz)
        time.sleep(1)
    visualizer.wait_for_window_closure()

    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(window_name="visualization", width=800, height=600)
    # options = visualizer.get_render_option()
    # options.background_color = (0.8, 0.8, 1.0)
    # options.light_on = True
    # visualizer.add_geometry(pcd)
    # visualizer.poll_events()
    # visualizer.update_renderer()

    # time.sleep(3)
    # visualizer.destroy_window()

    # Convert Open3D.o3d.geometry.PointCloud to numpy array.
    xyz_converted = np.asarray(pcd.points)
    print("Printing numpy array made using Open3D pointcloud ...")
    print(xyz_converted)


