import numpy as np


class CurveFromThreshold:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.threshold = threshold

    def extract_points(self, v_disparity):
        indexes_of_points = np.where(v_disparity > self.threshold)
        points_d = []
        for x in indexes_of_points[1]:
            points_d.append((x + 0.5) * self.bucket_size)
        return indexes_of_points[0], points_d

    def extract_points_indexes(self, v_disparity):
        indexes_of_points = np.where(v_disparity > self.threshold)
        return indexes_of_points[0], indexes_of_points[1]

    def get_curve(self, v_disparity):
        points_h, points_d = self.extract_points(v_disparity)
        return np.polyfit(points_h, points_d, deg=self.curve_degree)
