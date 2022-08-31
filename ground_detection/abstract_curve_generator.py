import numpy as np


class AbstractCurveGenerator:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.threshold = threshold

    def extract_points(self, v_disparity) -> tuple:
        pass

    def extract_points_indexes(self, v_disparity) -> tuple:
        pass

    def get_curve(self, v_disparity, full_data=False):
        points_h, points_d = self.extract_points(v_disparity)
        return np.polyfit(points_h, points_d, deg=self.curve_degree, full=full_data)
