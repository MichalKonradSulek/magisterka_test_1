import numpy as np


class CurveFromMax:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.threshold = threshold

    def extract_points(self, v_disparity):
        points_d = []
        points_h = []
        unfiltered_h = np.argmax(v_disparity, axis=0)
        for x in range(v_disparity.shape[1]):
            if v_disparity[unfiltered_h[x], x] > self.threshold:
                points_d.append((x + 0.5) * self.bucket_size)
                points_h.append(unfiltered_h[x])
        return points_h, points_d

    def extract_points_indexes(self, v_disparity):
        points_x = []
        points_y = []
        unfiltered_y = np.argmax(v_disparity, axis=0)
        for x in range(v_disparity.shape[1]):
            if v_disparity[unfiltered_y[x], x] > self.threshold:
                points_x.append(x)
                points_y.append(unfiltered_y[x])
        return points_x, points_y

    def get_curve(self, v_disparity):
        points_h, points_d = self.extract_points(v_disparity)
        return np.polyfit(points_h, points_d, deg=self.curve_degree)
