import numpy as np


class CurveFromLowest:
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree,
                 depth_multiplier_for_curve_generation=None):
        self.threshold = threshold
        self.bucket_size = max_depth / v_disparity_shape[1]
        self.curve_degree = curve_degree
        self.depth_multiplier_for_curve_generation = depth_multiplier_for_curve_generation

    def extract_points(self, v_disparity):
        points_d = []
        points_h = []
        for x in range(v_disparity.shape[1]):
            for y in range(v_disparity.shape[0] - 1, -1, -1):
                if v_disparity[y, x] > self.threshold:
                    points_d.append((x + 0.5) * self.bucket_size)
                    points_h.append(y)
                    break
        return points_h, points_d

    def extract_points_indexes(self, v_disparity):
        points_x = []
        points_y = []
        for x in range(v_disparity.shape[1]):
            for y in range(v_disparity.shape[0] - 1, -1, -1):
                if v_disparity[y, x] > self.threshold:
                    points_x.append(x)
                    points_y.append(y)
                    break
        return points_x, points_y

    def get_curve(self, v_disparity):
        points_h, points_d = self.extract_points(v_disparity)
        if self.depth_multiplier_for_curve_generation is None:
            return np.polyfit(points_h, points_d, deg=self.curve_degree)
        else:
            modified_d = [d*self.depth_multiplier_for_curve_generation for d in points_d]
            curve = np.polyfit(points_h, modified_d, deg=self.curve_degree)
            return [d/self.depth_multiplier_for_curve_generation for d in curve]
