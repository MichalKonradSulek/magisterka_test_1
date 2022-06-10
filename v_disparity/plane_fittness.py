import numpy as np


def _get_line_abc(line_a_b):
    return line_a_b[0], -1.0, line_a_b[1]


def _create_y_index_array(shape):
    column = np.arange(start=0, stop=shape[0])
    result = np.tile(column.reshape(-1, 1), (1, shape[1]))
    return result


# problem - współrzędna y nie jest w metrach...
class PlanePointsQualifier:
    def __init__(self, frame_shape, tolerance):
        self.y_index_array = _create_y_index_array(frame_shape)
        self.tolerance_square = tolerance * tolerance

    def _get_square_distance_from_plane(self, depth_frame, line_a_b):
        a, b, c = _get_line_abc(line_a_b)
        # a is depth multiplier, b is height multiplier
        denominator = a * a + b * b
        result = depth_frame * a + self.y_index_array * b + c
        result = result * result / denominator
        return result

    # line has equation y = a * depth + b
    def get_plane_pixels(self, depth_frame, line_a_b):
        square_distance = self._get_square_distance_from_plane(depth_frame, line_a_b)
        return square_distance < self.tolerance_square

    def get_plane_pixels_horizontal_tolerance(self, depth_image, line_a_b):
        plane = (self.y_index_array - line_a_b[1]) / line_a_b[0]
        horizontal_distance_to_plane = depth_image - plane
        pixels_on_plane = horizontal_distance_to_plane * horizontal_distance_to_plane < self.tolerance_square
        return pixels_on_plane


if __name__ == '__main__':
    depth_array = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 5.0, 5.0]
    ])
    line = (1, -2)
    qualifier = PlanePointsQualifier(frame_shape=depth_array.shape, tolerance=0.1)
    points_on_line = qualifier.get_plane_pixels(depth_array, line)
    print(points_on_line)
