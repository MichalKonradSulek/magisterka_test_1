import os
import cv2
import numpy as np

from monodepth2_runner import Monodepth2Runner
from u_disparity.u_disparity import UDisparityCalculator
from v_disparity.v_disparity import VDisparityCalculator
from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.curve_from_max import CurveFromMax
from ground_detection.ground_points_qualifier import _create_y_index_pows_column
from ground_detection.ground_points_qualifier import GroundPointsQualifier
from utilities.image_window_controller import ImageWindowController


def get_paths_to_all_valid_pictures(folder):
    paths = []
    files_in_dir = os.listdir(folder)
    for file in files_in_dir:
        name_and_extension = os.path.splitext(file)
        if name_and_extension[1] == '.png' and name_and_extension[0].endswith('_raw'):
            paths.append(os.path.join(folder, file))
    return paths


def transform_depth(depth_frame, a, b, c):
    result = depth_frame * depth_frame * a + depth_frame * b + c
    return result


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_scian"
    max_depth = 20
    disparity_levels = 100

    depth_generator = Monodepth2Runner()
    u_disparity_calculator = UDisparityCalculator(depth_generator.frame_shape, disparity_levels, max_depth)
    v_disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, disparity_levels, max_depth)
    u_disparity_window = ImageWindowController("u_disparity", resize_factor_x_y=(1, 2), run_cv_waitkey=False)
    v_disparity_window = ImageWindowController("v_disparity", resize_factor_x_y=(2, 2))
    paths_to_pictures = get_paths_to_all_valid_pictures(dir_path)

    for path in paths_to_pictures:
        bgr_image = cv2.imread(path)
        resized_bgr_image = cv2.resize(bgr_image, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
        transformed_depth = transform_depth(depth, -0.05, 1, 0)
        u_disparity = u_disparity_calculator.create_u_disparity(transformed_depth)
        v_disparity = v_disparity_calculator.create_v_disparity(transformed_depth)
        cv2.imshow("raw", resized_bgr_image)
        cv2.imshow("depth", depth / 20)
        u_disparity_window.show_image(u_disparity)
        v_disparity_window.show_image(v_disparity)
