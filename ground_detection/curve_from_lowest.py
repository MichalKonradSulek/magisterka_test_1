from ground_detection.abstract_curve_generator import AbstractCurveGenerator


class CurveFromLowest(AbstractCurveGenerator):

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
        return points_y, points_x
