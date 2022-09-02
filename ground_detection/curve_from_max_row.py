import numpy as np
from ground_detection.abstract_curve_generator import AbstractCurveGenerator


class CurveFromMaxRow(AbstractCurveGenerator):

    def extract_points(self, v_disparity):
        points_d = []
        points_h = []
        unfiltered_x = np.argmax(v_disparity, axis=1)
        for y in range(v_disparity.shape[0]):
            if v_disparity[y, unfiltered_x[y]] > self.threshold:
                points_d.append(self.d_index_row[unfiltered_x[y]])
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
