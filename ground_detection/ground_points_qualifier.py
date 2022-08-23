import numpy as np


def _create_y_index_pows_column(height, degree):
    column = np.arange(start=0, stop=height).reshape((-1, 1))
    pow_column = np.ones((height, 1))
    result = np.ones((height, 1))
    for deg in range(degree):
        pow_column = pow_column * column
        result = np.concatenate((pow_column, result), axis=1)
    return result


class GroundPointsQualifier:
    def __init__(self, frame_shape, tolerance, curve_degree):
        self.frame_shape = frame_shape
        self.y_index_pows_column = _create_y_index_pows_column(frame_shape[0], curve_degree)
        self.tolerance = tolerance

    def _create_curve_column(self, curve):
        coefficients_multiplied_by_y = self.y_index_pows_column * curve
        return coefficients_multiplied_by_y.sum(axis=1)

    def get_floor_pixels(self, depth_frame, curve):
        curve_array = np.tile(self._create_curve_column(curve).reshape((-1, 1)), (1, self.frame_shape[1]))
        return np.logical_and(curve_array - self.tolerance <= depth_frame,
                              depth_frame <= curve_array + self.tolerance)


if __name__ == '__main__':
    qualifier = GroundPointsQualifier((10, 5), 500, 3)
    r = qualifier.get_floor_pixels(np.ones((10, 5)) * 2000, (1, 10, 100, 1000))
    print(r)
