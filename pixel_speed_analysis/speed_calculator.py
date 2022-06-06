import numpy as np


def _calculate_a_coefficient(t, n):
    return 12.0 / (t * n * (n*n - 1))


def _calculate_b_coefficient(n):
    return (n - 1) / 2.0


def _create_c_coefficients_list(n):
    b = _calculate_b_coefficient(n)
    return [i - b for i in range(n)]


class SpeedCalculator:
    def __init__(self, frame_shape, input_video_fps=30, n_of_considered_frames=2):
        assert n_of_considered_frames >= 2
        assert input_video_fps > 0
        self.frame_shape = frame_shape
        self.n_of_frames = n_of_considered_frames
        self.frame_duration = 1.0 / input_video_fps
        self.a_coefficient = _calculate_a_coefficient(self.frame_duration, self.n_of_frames)
        self.c_coefficients_list = _create_c_coefficients_list(self.n_of_frames)
        self.array_of_depths = [np.zeros(self.frame_shape)] * self.n_of_frames
        self.mean_depth = None

    def _add_new_frame(self, new_frame):
        for i in range(self.n_of_frames - 1):
            self.array_of_depths[i] = self.array_of_depths[i+1]
        self.array_of_depths[-1] = new_frame

    def _calculate_mean_depth(self):
        sum_of_d = np.zeros(self.frame_shape)
        for frame in self.array_of_depths:
            sum_of_d += frame
        return sum_of_d / self.n_of_frames

    def get_speed(self, new_frame):
        self._add_new_frame(new_frame)
        mean_depth = self._calculate_mean_depth()
        temp_sum = np.zeros(self.frame_shape)
        for i in range(self.n_of_frames):
            temp_sum += self.c_coefficients_list[i] * (self.array_of_depths[i] - mean_depth)
        return temp_sum * self.a_coefficient
