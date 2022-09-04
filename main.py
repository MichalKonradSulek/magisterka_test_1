import cv2

from ground_detection.curve_from_lowest import CurveFromLowest
from ground_detection.ground_points_qualifier import GroundPointsQualifier
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator


def add_ground_to_picture(picture, ground):
    picture[:, :, 1][ground] += 50


if __name__ == '__main__':
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\slupek\\3_obok_nikon.MOV"

    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3
    threshold = 0.1
    tolerance = 0.7

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, v_disparity_levels, max_depth)
    curve_generator = CurveFromLowest(threshold, disparity_calculator.get_disparity_shape(), max_depth, curve_degree)
    ground_points_qualifier = GroundPointsQualifier(depth_generator.frame_shape, tolerance, curve_degree, max_depth)

    video = cv2.VideoCapture(video_path)

    success, frame = video.read()
    while success:
        resized_frame = cv2.resize(frame, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        curve = curve_generator.get_curve(v_disparity)
        ground_pixels = ground_points_qualifier.get_floor_pixels(depth, curve)

        add_ground_to_picture(resized_frame, ground_pixels)
        cv2.imshow("ground", resized_frame)
        cv2.waitKey(1)

        success, frame = video.read()
