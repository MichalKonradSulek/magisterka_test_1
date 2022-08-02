import cv2
import numpy as np

from utilities.timer import MyTimer
from v_disparity.v_disparity import VDisparityCalculator
from video import Monodepth2VideoInterpreter
from v_disparity.k_ransac_line import KRansacLine
from v_disparity.least_squares_line_fit import fit_line
from v_disparity.plane_fittness import PlanePointsQualifier
from v_disparity.projected_depth_calculator import ProjectedDepthCalculator
from monodepth2_runner import Monodepth2Runner


if __name__ == '__main__':

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    sensor_height = 0.0024192
    focal_length = 0.00405

    video_provider = Monodepth2VideoInterpreter(video_path, depth_generator=Monodepth2Runner())
    disparity_calculator = VDisparityCalculator(video_provider.frame_shape)
    v_disparity_threshold = 0.15
    line_generator = KRansacLine(k=150, point_threshold=v_disparity_threshold)
    plane_pixels_qualifier = PlanePointsQualifier(frame_shape=video_provider.frame_shape, tolerance=1.0)

    timer = MyTimer()
    timer.start()

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        v_disparity = disparity_calculator.create_v_disparity(depth_frame)
        timer.end_period("v-disparity")

        frame_height = v_disparity.shape[0]
        bottom_half_of_v_disparity = np.zeros(v_disparity.shape)
        bottom_half_of_v_disparity[int(frame_height / 2):frame_height, ...] = v_disparity[
                                                                              int(frame_height / 2):frame_height, ...]
        best_line = fit_line(bottom_half_of_v_disparity, point_threshold=v_disparity_threshold)
        timer.end_period("line")

        pixels_on_plane = plane_pixels_qualifier.get_plane_pixels_horizontal_tolerance(depth_frame, best_line)
        timer.end_period("pixels_on_plane")

        depth_to_show = depth_frame / 20
        depth_to_show = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        green_plane = depth_to_show[:, :, 1]
        green_plane[pixels_on_plane] = 255
        cv2.imshow("depth", depth_to_show)

        v_disparity_to_show = v_disparity * 4
        # v_disparity_to_show = np.zeros(v_disparity.shape)
        # condition = v_disparity > v_disparity_threshold
        # v_disparity_to_show[condition] = 255

        v_disparity_to_show = np.dstack((v_disparity_to_show, v_disparity_to_show, v_disparity_to_show))
        x0 = 0
        x1 = v_disparity_to_show.shape[1] - 1
        y0 = int(best_line[0] * x0 + best_line[1])
        y1 = int(best_line[0] * x1 + best_line[1])
        cv2.line(v_disparity_to_show, (x0, y0), (x1, y1), color=(0, 255, 0))
        v_disparity_to_show_size = (v_disparity.shape[1] * 2, v_disparity.shape[0] * 2)
        cv2.imshow("v-disparity", cv2.resize(v_disparity_to_show, v_disparity_to_show_size))
        cv2.waitKey(0)
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
