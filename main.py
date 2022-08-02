import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from video import Monodepth2VideoInterpreter
from visualizer.open3d_visualizer import Open3dVisualizer
from point_cloud_generation import generate_points_with_pix_coordinates
from point_cloud_generation import generate_3d_point_cloud
from clustering.dbscan_clustering import create_clusters
from cloud_generation.point_cloud_generator import PointCloudGenerator
from monodepth2_runner import Monodepth2Runner


def get_points_in_3d(array):
    width_pix = array.shape[1]
    height_pix = array.shape[0]
    mid_width_pix = (width_pix - 1) / 2.0
    mid_height_pix = (height_pix - 1) / 2.0
    matrix_width = 0.016
    matrix_height = 0.024
    pix_size_w = matrix_width / width_pix
    pix_size_h = matrix_height / height_pix
    f = 0.50
    x, y, z = [], [], []
    for i_h, i_w in itertools.product(range(height_pix), range(width_pix)):
        d = array[i_h, i_w]
        mx = (i_w - mid_width_pix) * pix_size_w
        my = (i_h - mid_height_pix) * pix_size_h
        md = math.sqrt(mx**2 + my**2 + f**2)
        k = d / md
        x.append(mx * k)
        z.append(-my * k)
        y.append(f * k)
    return x, y, z

def show_2d_plot(array):
    plt.imshow(array, interpolation='none')
    plt.show()


def show_3d_plot(x, y, z):
    fig = plt.figure()
    subplot = fig.add_subplot(projection='3d')
    subplot.set_xlabel('x')
    subplot.set_ylabel('y')
    subplot.set_zlabel('z')
    subplot.scatter(x, y, z, c=y, cmap='plasma_r', marker='.')
    plt.show()


def plot_clusters(list_of_clusters):
    clusters_fig = plt.figure()
    subplot = clusters_fig.add_subplot(projection='3d')
    subplot.set_xlabel('x')
    subplot.set_ylabel('y')
    subplot.set_zlabel('z')
    for cluster in list_of_clusters:
        subplot.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], marker='.')
    plt.show()


def plot_clusters_2d(list_of_clusters):
    fig = plt.figure()
    subplot = fig.add_subplot()
    subplot.set_xlabel('x')
    subplot.set_ylabel('z')
    for cluster in list_of_clusters:
        subplot.scatter(cluster[:, 0], cluster[:, 2], marker='.')
    plt.show()


def modify_points(points, width_modifier=1.0, height_modifier=1.0, depth_modifier=2.0):
    points[:, 0] = points[:, 0] * width_modifier
    points[:, 1] = points[:, 1] * depth_modifier
    points[:, 2] = points[:, 2] * height_modifier
    return points


if __name__ == '__main__':
    # print("test.py")
    # filename = "C:\\Users\\Michal\\Pictures\\Test\\DSC_0055_depth.npy"
    # depth_data = get_data_from_file(filename)
    # print(depth_data.shape)
    # # show_2d_plot(depth_data)
    # px, py, pz = get_points_coordinates(depth_data, step_width=5, step_height=4)
    # show_3d_plot(px, py, pz)
    # X = numpy.asarray(list(zip(px, py, pz)))
    # X = modify_points(X, width_modifier=1, height_modifier=1, depth_modifier=20)
    # clusters = create_clusters(X, eps=7.1, min_samples=5)
    # plot_clusters(clusters)
    # plot_clusters_2d(clusters)

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"
    video_provider = Monodepth2VideoInterpreter(video_path, depth_generator=Monodepth2Runner())
    cloud_generator = PointCloudGenerator(640, 192, 0.0043008, 0.0024192, 0.00405)
    points_visualizer = Open3dVisualizer(max_depth=20.0)

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        # xyz = generate_3d_point_cloud(depth_frame)
        # point_cloud = generate_points_with_pix_coordinates(depth_frame)
        # point_cloud = generate_3d_point_cloud(depth_frame, f=0.00405, pix_size=0.0000112)
        point_cloud = cloud_generator.generate(depth_frame)
        # points_visualizer.change_points(point_cloud)

        clusters = [point_cloud]
        # clusters = create_clusters(point_cloud, eps=3)
        points_visualizer.show_clouds(clusters)

        success, depth_frame = video_provider.get_next_depth_frame()
    points_visualizer.destroy_window()


