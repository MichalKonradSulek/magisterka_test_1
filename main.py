import itertools
import math

import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import DBSCAN


def get_data_from_file(filepath):
    print("file: " + filepath)
    original_data = numpy.load(filepath)
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


def create_clusters(points, eps: float = 9, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    # fit model and predict clusters
    yhat = model.fit_predict(points)
    # retrieve unique clusters
    clusters = numpy.unique(yhat)
    # create scatter plot for samples from each cluster
    result = []
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = numpy.where(yhat == cluster)
        # create scatter of these samples
        result.append(points[row_ix, :][0])
    return result


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
    print("test.py")
    filename = "C:\\Users\\Michal\\Pictures\\Test\\test7_depth.npy"
    depth_data = get_data_from_file(filename)
    print(depth_data.shape)
    # show_2d_plot(depth_data)
    px, py, pz = get_points_coordinates(depth_data, step_width=5, step_height=4)
    # show_3d_plot(px, py, pz)
    X = numpy.asarray(list(zip(px, py, pz)))
    X = modify_points(X, width_modifier=1, height_modifier=1, depth_modifier=20)
    clusters = create_clusters(X, eps=8.1, min_samples=5)
    plot_clusters(clusters)
    plot_clusters_2d(clusters)

