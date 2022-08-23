import numpy as np



class CurveFitting:
    def __init__(self, v_disparity_shape, max_depth, k_shuffles, threshold,
                 minimal_coefficients, maximal_coefficients):
        self.height = v_disparity_shape[0]
        self.width = v_disparity_shape[1]
        self.max_depth = max_depth
        self.k_shuffles = k_shuffles
        self.threshold = threshold
        self.minimal_coefficients = minimal_coefficients
        self.maximal_coefficient = maximal_coefficients
        assert len(minimal_coefficients) == len(maximal_coefficients)
        self.n_of_coefficients = len(minimal_coefficients)
        self.x_matrix = self.prepare_x_matrix()

    def prepare_x_matrix(self):
        base_numbers = np.arange(self.height).reshape((self.height, 1))
        column = base_numbers
        result = np.ones((self.height, 1))
        for i in range(self.n_of_coefficients - 1):
            result = np.concatenate((column, result), axis=1)
            column *= base_numbers
        return result

    def get_random_curve_coefficients(self):
        return np.random.uniform(self.minimal_coefficients, self.maximal_coefficient, self.k_shuffles)

    def get_best_curve(self, v_disparity):
        random_coefficients = self.get_random_curve_coefficients()
        for coefficients in random_coefficients:
            curve = np.sum(self.x_matrix * coefficients, axis=1)
            min_index = ((curve - self.threshold) / self.max_depth * self.width).astype('int').clip(min=0, max=self.width)
            max_index = ((curve + self.threshold) / self.max_depth * self.width).astype('int').clip(min=0, max=self.width)
            for y in range(self.height):




if __name__ == '__main__':
    base_numbers = np.arange(10).reshape((10, 1))
    column = base_numbers
    result = np.ones((10, 1))
    for i in range(2):
        result = np.concatenate((column, result), axis=1)
        column *= base_numbers

    print(result)
    print(np.sum(result, axis=1))
