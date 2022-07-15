import numpy as np


def find_the_furthest_one(column):
    line_values = np.linspace(column[0], column[-1], num=len(column))
    max_index = np.abs(column - line_values).argmax()
    return max_index


def calculate_distance_square(column, point_index):
    y1 = column[0]
    y2 = column[-1]
    x1 = 0
    x2 = len(column) - 1
    x = point_index
    y = column[point_index]
    numerator = ((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) ** 2
    denominator = ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
    return numerator / denominator


def calculate_stixels_ends(column, result_column, threshold):
    # result column is array with booleans; set to True if end of stixel
    if len(column) >= 3:
        the_furthest_index = find_the_furthest_one(column)
        if calculate_distance_square(column, the_furthest_index) > threshold:
            result_column[the_furthest_index] = True
            calculate_stixels_ends(column[0:(the_furthest_index + 1)], result_column[0:(the_furthest_index + 1)], threshold)
            calculate_stixels_ends(column[the_furthest_index:], result_column[the_furthest_index:], threshold)


def calculate_stixels_ends_for_frame(frame, threshold):
    result_array = np.full(frame.shape, False)
    for i in range(frame.shape[1]):
        calculate_stixels_ends(frame[:, i], result_array[:, i], threshold)
    return result_array


if __name__ == "__main__":
    array = np.array([[1.0, 2.0, 3.0, 1.0], [1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 3.0]])
    input_list = array[0, :]
    print(input_list)
    the_furthest = find_the_furthest_one(input_list)
    print(the_furthest)
    print(calculate_distance_square(input_list, the_furthest))

    array1 = np.linspace(0.0, 10.0, num=10, endpoint=False)
    array2 = np.linspace(10.0, -10.0, num=10, endpoint=False)
    array3 = np.linspace(-10.0, 1.0, num=12)
    array = np.concatenate((array1, array2, array3), axis=None)
    print(array)
    result = np.full(len(array), False)
    calculate_stixels_ends(array, result, 1.0)
    print(result)