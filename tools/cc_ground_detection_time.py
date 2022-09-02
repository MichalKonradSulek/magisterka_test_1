import os
import numpy as np
import cv2
import time

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


def print_result_array(array):
    for threshold in thresholds:
        print("\t%.2f" % threshold, end="")
    print()
    for row in range(array.shape[0]):
        print(generators_classes[row].__name__, end="")
        for column in range(array.shape[1]):
            print("\t%.3f" % array[row, column], end="")
        print()


if __name__ == "__main__":
    files_dir = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_podloza"
    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3
    threshold = 0.1
    number_of_runs = 1000

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)

    pairs_of_images = get_image_pairs_from_directory(files_dir)
    thresholds = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3]
    generators_classes = [CurveFromThreshold, CurveFromMaxRow, CurveFromMaxColumn, CurveFromLowest]
    times_for_generators = np.zeros((len(generators_classes), len(thresholds), len(pairs_of_images)))

    for i_pic in range(len(pairs_of_images)):
        original_path = pairs_of_images[i_pic][0]
        print(original_path)
        original = cv2.imread(original_path)
        resized_bgr_image = cv2.resize(original, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        for generator_i in range(len(generators_classes)):
            print(generators_classes[generator_i].__name__)
            for threshold_i in range(len(thresholds)):
                print("threshold", thresholds[threshold_i])
                curve_generator = generators_classes[generator_i](thresholds[threshold_i],
                                                                  (depth_generator.frame_shape[0], v_disparity_levels),
                                                                  max_depth, curve_degree)
                start_time = time.time()
                for i in range(number_of_runs):
                    ignored = curve_generator.get_curve(v_disparity)
                end_time = time.time()
                times_for_generators[generator_i, threshold_i, i_pic] = end_time - start_time
    result = times_for_generators.mean(axis=2)
    print_result_array(result)
