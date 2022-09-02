import numpy as np

from ground_detection.abstract_curve_generator import AbstractCurveGenerator


def create_index_array(shape):
    column = np.arange(shape[0]).reshape((-1, 1))
    array = np.tile(column, (1, shape[1]))
    return array


class CurveFromLowest(AbstractCurveGenerator):
    def __init__(self, threshold, v_disparity_shape, max_depth, curve_degree):
        AbstractCurveGenerator.__init__(self, threshold, v_disparity_shape, max_depth, curve_degree)
        self.y_index_array = create_index_array(v_disparity_shape)

    def extract_points(self, v_disparity):
        higher_than_threshold = v_disparity > self.threshold
        selected_points = np.ones(self.y_index_array.shape) * -1
        selected_points[higher_than_threshold] = self.y_index_array[higher_than_threshold]
        ys = selected_points.max(axis=0)
        good_points = ys >=0
        points_h = ys[good_points]
        points_d = self.d_index_row[good_points]
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


if __name__ == '__main__':
    a = np.array([[1, 0, 0],
                  [0, 4, 0],
                  [0, 0, 1]])
    for x in range(a.shape[1]):
        y = np.max(np.nonzero(a[:, x]))
        print(y)
    stamp = np.array([[True, False, False],
                      [False, False, False],
                      [False, False, True]])
    print(create_d_index_row(10, 5))
