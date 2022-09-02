import numpy as np
from ground_detection.abstract_curve_generator import AbstractCurveGenerator


class CurveFromMaxColumn(AbstractCurveGenerator):

    def extract_points(self, v_disparity):
        points_d = []
        points_h = []
        unfiltered_h = np.argmax(v_disparity, axis=0)
        for x in range(v_disparity.shape[1]):
            if v_disparity[unfiltered_h[x], x] > self.threshold:
                points_d.append(self.d_index_row[x])
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
        return points_y, points_x
