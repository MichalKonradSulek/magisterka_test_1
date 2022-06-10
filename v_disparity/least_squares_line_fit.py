import numpy as np


def extract_points(v_disparity_array, point_threshold):
    indexes_of_points = np.where(v_disparity_array > point_threshold)
    return indexes_of_points[1], indexes_of_points[0]


def fit_line(v_disparity, point_threshold):
    points_x, points_y = extract_points(v_disparity, point_threshold)
    result = np.polyfit(points_x, points_y, deg=1)
    return result[0], result[1]

