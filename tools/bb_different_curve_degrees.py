"""
Skrypt służy do analizy jakości krzywej wielomianowej różnego stopnia.
"""

import os
import time

import cv2
import numpy as np

from ground_detection.curve_from_threshold import CurveFromThreshold
from ground_detection.ground_points_qualifier import _create_y_index_pows_column
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator


def get_paths_to_all_valid_pictures(folder):
    paths = []
    files_in_dir = os.listdir(folder)
    for file in files_in_dir:
        name_and_extension = os.path.splitext(file)
        if name_and_extension[1] == '.png' and not name_and_extension[0].endswith('_result'):
            paths.append(os.path.join(folder, file))
    return paths


def create_curve_mask(curve, shape, maximum_depth):
    y_index_pows = _create_y_index_pows_column(shape[0], len(curve) - 1)
    products = y_index_pows * curve
    depths = products.sum(axis=1)
    x_indexes = (depths / maximum_depth * shape[1] - 0.5).astype('int')
    result = np.full(shape, False)
    for y in range(shape[0]):
        if 0 <= x_indexes[y] < shape[1]:
            result[y, x_indexes[y]] = True
    return result


def prepare_result_for_curve_generator(disparity, curve_generator):
    result = curve_generator.get_curve(disparity, full_data=True)
    curve = result[0]
    residuals = result[1][0]

    disparity_to_show = 3 * disparity
    disparity_rgb = np.dstack((disparity_to_show, disparity_to_show, disparity_to_show))
    curve_mask = create_curve_mask(curve, disparity.shape, max_depth)
    disparity_rgb[:, :, 0][curve_mask] = 0
    disparity_rgb[:, :, 1][curve_mask] = 0
    disparity_rgb[:, :, 2][curve_mask] = 1

    double_size = (int(disparity.shape[1] * 2), int(disparity.shape[0] * 2))
    disparity_rgb = cv2.resize(disparity_rgb, double_size, interpolation=cv2.INTER_NEAREST)
    disparity_rgb = (disparity_rgb * 255).clip(min=0, max=255).astype('uint8')

    return disparity_rgb, residuals


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\pusty_chodnik"
    max_depth = 20
    v_disparity_levels = 100
    threshold = 0.1

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    curve_generator_1 = CurveFromThreshold(threshold, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, 1)
    curve_generator_2 = CurveFromThreshold(threshold, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, 2)
    curve_generator_3 = CurveFromThreshold(threshold, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, 3)
    curve_generator_4 = CurveFromThreshold(threshold, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, 4)
    paths_to_pictures = get_paths_to_all_valid_pictures(dir_path)

    # Ta część służy do generowania wyniku w postaci obrazka oraz dostarcza informację o dokładności każdego rozwiązania
    # for path in paths_to_pictures:
    #     bgr_image = cv2.imread(path)
    #     resized_bgr_image = cv2.resize(bgr_image, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
    #     depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
    #     v_disparity = disparity_calculator.create_v_disparity(depth)
    #     v_d_1, residuals_1 = prepare_result_for_curve_generator(v_disparity, curve_generator_1)
    #     v_d_2, residuals_2 = prepare_result_for_curve_generator(v_disparity, curve_generator_2)
    #     v_d_3, residuals_3 = prepare_result_for_curve_generator(v_disparity, curve_generator_3)
    #     v_d_4, residuals_4 = prepare_result_for_curve_generator(v_disparity, curve_generator_4)
    #     original_new_size = (int(v_d_1.shape[0] * bgr_image.shape[1] / bgr_image.shape[0]), v_d_1.shape[0])
    #     resized_original = cv2.resize(bgr_image, original_new_size)
    #     white_bar = np.ones((v_d_1.shape[0], 2, 3), dtype='uint8') * 255
    #     picture_to_show = np.concatenate((resized_original, white_bar,  v_d_1,  white_bar, v_d_2, white_bar, v_d_3, white_bar, v_d_4), axis=1)
    #
    #     print(os.path.basename(path), residuals_1, residuals_2, residuals_3, residuals_4, sep='\t')
    #     base_and_extension = os.path.splitext(path)
    #     cv2.imwrite(base_and_extension[0] + '_result' + base_and_extension[1], picture_to_show)
    #
    #     cv2.imshow("result", picture_to_show)
    #     cv2.waitKey(0)

    # Ta część służy do badania czasu wykonywania operacji
    array_of_generators = [curve_generator_1, curve_generator_2, curve_generator_3, curve_generator_4]
    run_test_n_times = 1
    run_each_n_times = 1000
    result_table = np.zeros((run_test_n_times, len(array_of_generators), len(paths_to_pictures)))
    for test_i in range(run_test_n_times):
        print("test:", test_i)
        for picture_i in range(len(paths_to_pictures)):
            print(os.path.basename(paths_to_pictures[picture_i]), end=" | ")
            bgr_image = cv2.imread(paths_to_pictures[picture_i])
            resized_bgr_image = cv2.resize(bgr_image, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
            depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
            v_disparity = disparity_calculator.create_v_disparity(depth)
            for generator_i in range(len(array_of_generators)):
                print("generator", generator_i + 1, sep='_', end=' ')
                start_time = time.time()
                for i in range(run_each_n_times):
                    ignored = array_of_generators[generator_i].get_curve(v_disparity)
                end_time = time.time()
                result_table[test_i, generator_i, picture_i] = end_time - start_time
            print()
    mean_result = result_table.mean(axis=2)
    print(mean_result)
