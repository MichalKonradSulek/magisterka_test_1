import cv2
import numpy as np

from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.curve_from_max_column import CurveFromMaxColumn
from ground_detection.ground_points_qualifier import GroundPointsQualifier
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
from utilities.image_window_controller import ImageWindowController
from ground_detection.ground_points_qualifier import _create_y_index_pows_column



def add_ground_to_picture(picture, ground):
    picture[:, :, 1][ground] += 50


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


if __name__ == '__main__':
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\kosz\\1_obok_nikon.MOV"

    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3
    threshold = 0.1
    tolerance = 0.7

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    curve_generator = CurveFromMaxColumn(threshold, disparity_calculator.get_disparity_shape(), max_depth, curve_degree)
    ground_points_qualifier = GroundPointsQualifier(depth_generator.frame_shape, tolerance, curve_degree, max_depth)
    image_window = ImageWindowController("result", run_cv_waitkey=False, resize_factor_x_y=(1, 2))
    depth_window = ImageWindowController("depth", run_cv_waitkey=False, resize_factor_x_y=(1, 2))

    video = cv2.VideoCapture(video_path)

    success, frame = video.read()
    while success:
        resized_frame = cv2.resize(frame, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        curve = curve_generator.get_curve(v_disparity)
        ground_pixels = ground_points_qualifier.get_floor_pixels(depth, curve)

        add_ground_to_picture(resized_frame, ground_pixels)
        v_disparity_with_curve = get_v_disparity_with_curve(v_disparity, curve)
        # cv2.imshow("ground", resized_frame)
        image_window.show_image(resized_frame)
        depth_window.show_image(depth / 20)
        cv2.imshow("v-disparity", v_disparity_with_curve)
        cv2.waitKey(0)

        success, frame = video.read()
