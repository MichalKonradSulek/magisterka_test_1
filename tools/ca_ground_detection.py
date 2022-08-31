import os
import cv2
import numpy as np

from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.curve_from_max_column import CurveFromMaxColumn
from ground_detection.ground_points_qualifier import _create_y_index_pows_column
from ground_detection.ground_points_qualifier import GroundPointsQualifier
from utilities.image_window_controller import ImageWindowController


def get_paths_to_all_valid_pictures(folder):
    paths = []
    files_in_dir = os.listdir(folder)
    for file in files_in_dir:
        name_and_extension = os.path.splitext(file)
        if name_and_extension[1] == '.png' and not name_and_extension[0].endswith('_pd'):
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


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\wykrywanie_podloza"
    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    # curve_generator = CurveFromLowest(0.1, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, curve_degree)
    curve_generator = CurveFromMaxColumn(0.1, (depth_generator.frame_shape[0], v_disparity_levels), max_depth, curve_degree)
    ground_points_qualifier = GroundPointsQualifier(depth_generator.frame_shape, 0.7, curve_degree)
    disparity_window = ImageWindowController("disparity", resize_factor_x_y=(2, 2))
    paths_to_pictures = get_paths_to_all_valid_pictures(dir_path)

    for path in paths_to_pictures:
        bgr_image = cv2.imread(path)
        resized_bgr_image = cv2.resize(bgr_image, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        curve = curve_generator.get_curve(v_disparity)
        ground_pixels = ground_points_qualifier.get_floor_pixels(depth, curve)
        far_objects = depth > max_depth
        resized_bgr_image[:, :, 0][ground_pixels] = 255
        resized_bgr_image[:, :, 2][far_objects] = 255
        cv2.imshow("ground", resized_bgr_image)
        cv2.imshow("depth", depth / 20)

        v_disparity_to_show = v_disparity
        v_disparity_rgb = np.dstack((v_disparity_to_show, v_disparity_to_show, v_disparity_to_show))
        curve_mask = create_curve_mask(curve, v_disparity.shape, max_depth)
        v_disparity_rgb[:, :, 0][curve_mask] = 0
        v_disparity_rgb[:, :, 1][curve_mask] = 1
        v_disparity_rgb[:, :, 2][curve_mask] = 0
        disparity_window.show_image(v_disparity_rgb)



