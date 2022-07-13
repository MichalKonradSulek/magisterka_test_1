from sklearn.linear_model import LinearRegression
import numpy as np


class DepthEqualizer:
    def __init__(self, n_of_considered_frames):
        self.regression_n = n_of_considered_frames
        self.remembered_means = [0.0] * self.regression_n
        self.regression_x_array = np.array(range(self.regression_n)).reshape((-1, 1))

    def _move_all_remembered_elements(self):
        for j in range(len(self.remembered_means) - 1):
            self.remembered_means[j] = self.remembered_means[j+1]

    def _get_equalized_mean(self, current_mean):
        self._move_all_remembered_elements()
        self.remembered_means[-1] = current_mean
        regression = LinearRegression().fit(self.regression_x_array, self.remembered_means)
        return regression.predict([[self.regression_n - 1]])

    def get_equalized_depth(self, depth_frame):
        current_mean = depth_frame.mean()
        mean_from_regression = self._get_equalized_mean(current_mean)
        difference = mean_from_regression - current_mean
        return depth_frame + difference
