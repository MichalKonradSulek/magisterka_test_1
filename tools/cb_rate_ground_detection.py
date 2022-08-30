import os
import numpy as np
import cv2

from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.curve_from_max import CurveFromMax
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
    return result


def compare_results(from_analysis, ground_true, depth, count_up_to_distance=20.0):
    close_pixels = depth <= count_up_to_distance
    ground_pixels = np.logical_and(ground_true[:, :, 1] > 0, close_pixels)
    obstacles_pixels = np.logical_and(ground_true[:, :, 2] > 0, close_pixels)
    close_from_analysis = np.logical_and(from_analysis, close_pixels)
    not_found_ground = np.logical_and(ground_pixels, ~close_from_analysis)
    qualified_obstacles = np.logical_and(close_from_analysis, obstacles_pixels)
    return not_found_ground, qualified_obstacles, ground_pixels, obstacles_pixels


def print_results(not_found_ground, qualified_obstacles, ground_pixels, obstacles_pixels, picture_name):
    n_of_ground = np.count_nonzero(ground_pixels)
    n_of_obstacles = np.count_nonzero(obstacles_pixels)
    percent_not_found_ground = np.count_nonzero(not_found_ground) / n_of_ground * 100
    percent_qualified_obstacles = np.count_nonzero(qualified_obstacles) / n_of_obstacles * 100
    print("%s, %.2f, %.2f, %d, %d" % (picture_name, percent_not_found_ground, percent_qualified_obstacles, n_of_ground, n_of_obstacles))


if __name__ == "__main__":
    files_dir = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_podloza"
    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    curve_generator = CurveFromMax(0.1, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, curve_degree)
    ground_points_qualifier = GroundPointsQualifier(depth_generator.frame_shape, 0.7, curve_degree)
    result_window = ImageWindowController(resize_factor_x_y=(2, 2))

    pairs_of_images = get_image_pairs_from_directory(files_dir)
    print("not found ground, qualified obstacles, ground_pixels, obstacles_pixels")
    for original_path, ground_true_path in pairs_of_images:
        original = cv2.imread(original_path)
        resized_bgr_image = cv2.resize(original, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        curve = curve_generator.get_curve(v_disparity)
        ground_pixels = ground_points_qualifier.get_floor_pixels(depth, curve)

        true_picture = cv2.imread(ground_true_path)
        true_picture_resized = cv2.resize(true_picture, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        results = compare_results(ground_pixels, true_picture_resized, depth)
        presented_result = get_presented_result(resized_bgr_image, results[0], results[1])
        picture_name = os.path.basename(original_path)
        print_results(results[0], results[1], results[2], results[3], picture_name)
        result_window.show_image(presented_result)
