import numpy
import matplotlib.pyplot as plt
import itertools
import math


def get_data_from_file(filepath):
    print("file: " + filepath)
    original_data = numpy.load(filepath)
    return original_data[0, 0, :, :]


def get_points_coordinates(array, step: int = 1):
    x = []
    y = []
    z = []
    width = array.shape[1]
    height = array.shape[0]
    for v, h in itertools.product(range(0, height, step), range(0, width, step)):
        x.append(h)
        z.append(height - v)
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


def show_3d_plot(array, scatter: int = 1):
    px, py, pz = get_points_coordinates(array, step=scatter)
    # px, py, pz = get_points_in_3d(array)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(px, py, pz, c=py, cmap='plasma_r', marker='.')
    plt.show()


if __name__ == '__main__':
    print("test.py")
    filename = "C:/Users/Michal/PycharmProjects/monodepth2/assets/DSC_1838_depth.npy"
    depth_data = get_data_from_file(filename)
    print(depth_data.shape)
    show_2d_plot(depth_data)
    show_3d_plot(depth_data, scatter=8)

