import numpy as np


def get_plane_pixels(depth_image, line, distance_to_plane):
    planes_column = np.zeros((depth_image.shape[0], 1))
    for y in range(depth_image.shape[0]):
        planes_column[y] = (float(y) - line[1]) / line[0]
    plane = np.tile(planes_column, (1, depth_image.shape[1]))
    difference = depth_image - plane
    threshold = distance_to_plane * distance_to_plane
    pixels_on_plane = difference * difference < threshold
    return pixels_on_plane
