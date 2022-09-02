import numpy as np


def create_d_index_row(x_size, bucket_size):
    row = np.arange(x_size, dtype='float')
    return (row + 0.5) * bucket_size


class AbstractCurveGenerator:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.threshold = threshold
        self.d_index_row = create_d_index_row(v_disparity_shape[1], self.bucket_size)

    def extract_points(self, v_disparity) -> tuple:
        pass

    def extract_points_indexes(self, v_disparity) -> tuple:
        pass

    def get_curve(self, v_disparity, full_data=False):
        points_h, points_d = self.extract_points(v_disparity)
        return np.polyfit(points_h, points_d, deg=self.curve_degree, full=full_data)
