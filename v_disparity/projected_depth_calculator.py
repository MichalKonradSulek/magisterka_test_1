import numpy as np


# Calculates depth projected on vertical axis
class ProjectedDepthCalculator:
    def __init__(self, frame_shape, sensor_height, focal_length):
        self.frame_shape = frame_shape
        self.pixel_height = sensor_height / frame_shape[0]
        self.focal_length = focal_length

        self.depth_multipliers_matrix = self._create_depth_multipliers_matrix()

    def _create_depth_multipliers_matrix(self):
        column = np.arange(start=0.0, stop=self.frame_shape[0], dtype=np.float32)
        column = column - (self.frame_shape[0] - 1) / 2.0
        column *= self.pixel_height
        pixel_height_array = np.tile(column.reshape(-1, 1), (1, self.frame_shape[1]))
        denominator = np.sqrt(pixel_height_array * pixel_height_array + self.focal_length * self.focal_length)
        result = self.focal_length / denominator
        return result

    def get_depth_projected_onto_horizontal_plane(self, depth_frame):
        return depth_frame * self.depth_multipliers_matrix


if __name__ == '__main__':
    a = ProjectedDepthCalculator((5, 4), 1.0, 1.0)
    print(a.depth_multipliers_matrix)
