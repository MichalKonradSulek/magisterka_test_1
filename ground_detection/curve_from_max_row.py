import numpy as np


class CurveFromMaxRow:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.threshold = threshold

    def extract_points(self, v_disparity):
        points_d = []
        points_h = []
        unfiltered_h = np.argmax(v_disparity, axis=1)
        for y in range(v_disparity.shape[0]):
            if v_disparity[y, unfiltered_h[y]] > self.threshold:
                points_d.append((unfiltered_h[y] + 0.5) * self.bucket_size)
                points_h.append(y)
        return points_h, points_d

    def extract_points_indexes(self, v_disparity):
        points_x = []
        points_y = []
        unfiltered_x = np.argmax(v_disparity, axis=1)
        for y in range(v_disparity.shape[0]):
            if v_disparity[y, unfiltered_x[y]] > self.threshold:
                points_x.append(unfiltered_x[y])
                points_y.append(y)
        return points_y, points_x

    def get_curve(self, v_disparity):
        points_h, points_d = self.extract_points(v_disparity)
        return np.polyfit(points_h, points_d, deg=self.curve_degree)
