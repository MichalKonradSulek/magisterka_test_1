import os
import numpy as np
import cv2

from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.curve_from_max_column import CurveFromMaxColumn
from ground_detection.curve_from_max_row import CurveFromMaxRow
from ground_detection.curve_from_threshold import CurveFromThreshold
from ground_detection.ground_points_qualifier import _create_y_index_pows_column
from ground_detection.ground_points_qualifier import GroundPointsQualifier
from utilities.image_window_controller import ImageWindowController


def get_image_pairs_from_directory(directory_path):
    files_in_dir = os.listdir(directory_path)
    qualified_files = []
    for item in files_in_dir:
        path = os.path.join(directory_path, item)
        if os.path.isfile(path):
            filename_and_extension = os.path.splitext(item)
            if filename_and_extension[1] == ".png" and not filename_and_extension[0].endswith("_true"):
                potential_true_file = os.path.join(directory_path,
                                                   filename_and_extension[0] + '_true' + filename_and_extension[1])
                if os.path.exists(potential_true_file):
                    qualified_files.append((path, potential_true_file))
    return qualified_files


def get_presented_result(resized_bgr, not_found_ground, qualified_obstacles):
    multiplication = 2.0
    addition = 20
    result = np.copy(resized_bgr)
    result[:, :, 0][not_found_ground] = ((result[:, :, 0][not_found_ground] + addition) * multiplication) \
        .clip(max=255).astype('uint8')
    result[:, :, 2][qualified_obstacles] = ((result[:, :, 2][qualified_obstacles] + addition) * multiplication) \
        .clip(max=255).astype('uint8')
    result_desired_size = (
    result.shape[1], int(result.shape[1] * original.shape[0] / original.shape[1]))
    result = cv2.resize(result, result_desired_size)
    return result


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


def get_v_disparity_with_curve(disparity, curve):
    disparity_to_show = 3 * disparity
    disparity_rgb = np.dstack((disparity_to_show, disparity_to_show, disparity_to_show))
    curve_mask = create_curve_mask(curve, disparity.shape, max_depth)
    disparity_rgb[:, :, 0][curve_mask] = 0
    disparity_rgb[:, :, 1][curve_mask] = 0
    disparity_rgb[:, :, 2][curve_mask] = 1
    double_size = (int(disparity.shape[1] * 2), int(disparity.shape[0] * 2))
    disparity_rgb = cv2.resize(disparity_rgb, double_size, interpolation=cv2.INTER_NEAREST)
    disparity_rgb = (disparity_rgb * 255).clip(min=0, max=255).astype('uint8')
    return disparity_rgb


def get_v_disparity_with_selected_points(disparity, generator):
    disparity_to_show = 3 * disparity
    disparity_rgb = np.dstack((disparity_to_show, disparity_to_show, disparity_to_show))
    points_y, points_x = generator.extract_points_indexes(disparity)
    for y, x in zip(points_y, points_x):
        disparity_rgb[y, x, 0] = 1
        disparity_rgb[y, x, 1] = 0
        disparity_rgb[y, x, 2] = 1
    double_size = (int(disparity.shape[1] * 2), int(disparity.shape[0] * 2))
    disparity_rgb = cv2.resize(disparity_rgb, double_size, interpolation=cv2.INTER_NEAREST)
    disparity_rgb = (disparity_rgb * 255).clip(min=0, max=255).astype('uint8')
    return disparity_rgb


def save_pictures(result, with_curve, with_points, generator_name, threshold):
    base_name = os.path.basename(original_path)
    name_and_extension = os.path.splitext(base_name)
    new_name = name_and_extension[0] + '_' + generator_name + '_' + str(int(threshold * 100)).zfill(2) + ".png"
    result_path = os.path.join(os.path.join(result_dir, "obraz"), new_name)
    curve_path = os.path.join(os.path.join(result_dir, "krzywa"), new_name)
    points_path = os.path.join(os.path.join(result_dir, "punkty"), new_name)
    cv2.imwrite(result_path, result)
    cv2.imwrite(curve_path, with_curve)
    cv2.imwrite(points_path, with_points)


def compare_results(from_analysis, ground_true, depth, count_up_to_distance=20.0):
    close_pixels = depth <= count_up_to_distance
    ground_pixels = np.logical_and(ground_true[:, :, 1] > 0, close_pixels)
    obstacles_pixels = np.logical_and(ground_true[:, :, 2] > 0, close_pixels)
    close_from_analysis = np.logical_and(from_analysis, close_pixels)
    not_found_ground = np.logical_and(ground_pixels, ~close_from_analysis)
    qualified_obstacles = np.logical_and(close_from_analysis, obstacles_pixels)
    return not_found_ground, qualified_obstacles, ground_pixels, obstacles_pixels


def get_percentage(pixels, all_pixels):
    n_of_all = np.count_nonzero(all_pixels)
    percent_not_found_ground = np.count_nonzero(pixels) / n_of_all * 100
    return percent_not_found_ground


def print_results(not_found_ground, qualified_obstacles, ground_pixels, obstacles_pixels, picture_name, threshold, generator_name):
    n_of_ground = np.count_nonzero(ground_pixels)
    n_of_obstacles = np.count_nonzero(obstacles_pixels)
    percent_not_found_ground = np.count_nonzero(not_found_ground) / n_of_ground * 100
    percent_qualified_obstacles = np.count_nonzero(qualified_obstacles) / n_of_obstacles * 100
    print("%s\t%s\t%.2f\t%.2f\t%.2f\t%d\t%d" % (picture_name, generator_name, threshold, percent_not_found_ground, percent_qualified_obstacles, n_of_ground, n_of_obstacles))


def print_result_array(array):
    for threshold in thresholds:
        print("\t%.2f" % threshold, end="")
    print()
    for row in range(array.shape[0]):
        print(generators_classes[row].__name__, end="")
        for column in range(array.shape[1]):
            print("\t%.2f" % array[row, column], end="")
        print()


if __name__ == "__main__":
    files_dir = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_podloza"
    result_dir = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_podloza\\wyniki"
    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    ground_points_qualifier = GroundPointsQualifier(depth_generator.frame_shape, 0.7, curve_degree)
    # result_window = ImageWindowController(resize_factor_x_y=(2, 2))

    thresholds = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3]
    generators_classes = [CurveFromThreshold, CurveFromMaxRow, CurveFromMaxColumn, CurveFromLowest]
    not_found_ground_values = np.zeros((len(generators_classes), len(thresholds)))
    qualified_obstacles_values = np.zeros((len(generators_classes), len(thresholds)))


    pairs_of_images = get_image_pairs_from_directory(files_dir)
    print("not found ground, qualified obstacles, ground_pixels, obstacles_pixels")
    for original_path, ground_true_path in pairs_of_images:
        original = cv2.imread(original_path)
        resized_bgr_image = cv2.resize(original, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        true_picture = cv2.imread(ground_true_path)
        true_picture_resized = cv2.resize(true_picture, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        print(os.path.basename(original_path))
        for generator_i in range(len(generators_classes)):
            for threshold_i in range(len(thresholds)):
                curve_generator = generators_classes[generator_i](thresholds[threshold_i],
                                                                  (depth_generator.frame_shape[0], v_disparity_levels),
                                                                  max_depth, curve_degree)
                curve = curve_generator.get_curve(v_disparity)
                ground_pixels = ground_points_qualifier.get_floor_pixels(depth, curve)

                not_found, found_obst, ground, obst = compare_results(ground_pixels, true_picture_resized, depth)
                not_found_ground_values[generator_i, threshold_i] = get_percentage(not_found, ground)
                qualified_obstacles_values[generator_i, threshold_i] = get_percentage(found_obst, obst)

                picture_name = os.path.basename(original_path)
                presented_result = get_presented_result(resized_bgr_image, not_found, found_obst)
                disparity_with_curve = get_v_disparity_with_curve(v_disparity, curve)
                disparity_with_points = get_v_disparity_with_selected_points(v_disparity, curve_generator)
                # save_pictures(presented_result, disparity_with_curve, disparity_with_points, type(curve_generator).__name__, curve_generator.threshold)

                # cv2.imshow("curve", disparity_with_curve)
                # cv2.imshow("points", disparity_with_points)
                # result_window.show_image(presented_result)
        print("ground not found:")
        print_result_array(not_found_ground_values)
        print("obstacles qualified:")
        print_result_array(qualified_obstacles_values)
        print()
