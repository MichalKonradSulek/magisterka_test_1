import numpy as np


class IPM:
    """
    Klasa oblicza położenie każdego piksela obrazu w przestrzeni (rzut na płaszczyznę poziomą) zgodnie
    z artykułem GOLD_a_parallel_real-time_stereo_vision_system_for_generic_obstacle_and_lane_detection.pdf
    """
    def __init__(self, l, h, theta, angular_aperture, n, gamma=0, d=0):
        self._theta_minus_alpha = theta - angular_aperture / 2.0
        self._gamma_minus_alpha = gamma - angular_aperture / 2.0
        self._aa_div_n_minus_1 = angular_aperture / (n - 1)
        self._h = h
        self._l = l
        self._d = d
        self._n = n
        self._create_x_y_matrices()

    def _create_x_y_matrices(self):
        u_indices = np.tile(np.arange(self._n).reshape((self._n, 1)), (1, self._n))
        v_indices = np.tile(np.arange(self._n), (self._n, 1))

        first_bracket = u_indices * self._aa_div_n_minus_1 + self._theta_minus_alpha
        second_bracket = v_indices * self._aa_div_n_minus_1 + self._gamma_minus_alpha
        ctg = np.ones(u_indices.shape) / np.tan(first_bracket)
        cos = np.cos(second_bracket)
        sin = np.sin(second_bracket)
        self.x = ctg * cos * self._h + self._l
        self.y = ctg * sin * self._h + self._d

    def transform_frame(self, frame, v_pix, h_pix, scale=1):
        if frame.shape != (self._n, self._n):
            raise Exception("Frame shape mismatch")

        result = np.zeros((v_pix, h_pix), dtype=frame.dtype)
        for i in range(self._n):
            for j in range(self._n):
                v = int(v_pix - self.x[i, j] * scale)
                h = int(self.y[i, j] * scale + h_pix / 2)
                if 0 <= v < v_pix and 0 <= h < h_pix:
                    result[v, h] = max(result[v, h], frame[i, j])
        return result