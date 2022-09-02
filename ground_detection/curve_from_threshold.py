import numpy as np
from ground_detection.abstract_curve_generator import AbstractCurveGenerator


class CurveFromThreshold(AbstractCurveGenerator):

    def extract_points(self, v_disparity):
        indexes_of_points = np.where(v_disparity > self.threshold)
        points_d = (indexes_of_points[1].astype('float') + 0.5) * self.bucket_size
        return indexes_of_points[0], points_d

    def extract_points_indexes(self, v_disparity):
        indexes_of_points = np.where(v_disparity > self.threshold)
        return indexes_of_points[0], indexes_of_points[1]
