import sys

import numpy as np

from random import randint


def calculate_line_fitness(line, points):
    denominator = line[0] * line[0] + line[1] * line[1]
    result = np.multiply(points, [line[0], line[1]])
    result = result + line[2]
    result = result.sum(axis=1)
    result = result * result / denominator
    return result.sum()


def get_k_random_pairs_of_indexes(k, max_i_excluded):
    result = []
    for i in range(k):
        i1 = randint(0, max_i_excluded - 1)
        i2 = randint(0, max_i_excluded - 1)
        if i1 != i2:
            result.append((i1, i2))
        else:
            i -= 1
    return result


def create_line_from_two_points(p0, p1):
    a = float(p0[1] - p1[1])
    b = float(p1[0] - p0[0])
    c = float(p0[0] * p1[1] - p1[0] * p0[1])
    return a, b, c


class KRansacLine:
    def __init__(self, k, point_threshold=0.3):
        self.k = k
        self.point_threshold = point_threshold

    def extract_points(self, v_disparity_array):
        indexes_of_points = np.where(v_disparity_array > self.point_threshold)
        return np.stack((indexes_of_points[1], indexes_of_points[0]), axis=1)

    def calculate_line(self, v_disparity_array):
        best_line = None
        best_fitness = sys.float_info.max
        points = self.extract_points(v_disparity_array)
        print(points.shape)
        random_pairs = get_k_random_pairs_of_indexes(self.k, points.shape[0])
        for i0, i1 in random_pairs:
            line = create_line_from_two_points(points[i0], points[i1])
            line_fitness = calculate_line_fitness(line, points)
            if line_fitness < best_fitness:
                best_fitness = line_fitness
                best_line = line
        return best_line



