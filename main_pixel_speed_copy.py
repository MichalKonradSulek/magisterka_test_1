import cv2
import numpy as np

from pixel_speed_analysis.collision_detection import CollisionDetector
from pixel_speed_analysis.collision_detection import calculate_time_to_collision
from pixel_speed_analysis.speed_calculator import SpeedCalculator
from video import Monodepth2VideoInterpreter
from utilities.timer import MyTimer
from monodepth2_runner import Monodepth2Runner


if __name__ == '__main__':

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = Monodepth2VideoInterpreter(video_path, depth_generator=Monodepth2Runner())

    speed_calculator = SpeedCalculator(frame_shape=video_provider.frame_shape, n_of_considered_frames=30)
    speed_calculator_2 = SpeedCalculator(frame_shape=video_provider.frame_shape, n_of_considered_frames=2)
    collision_detector = CollisionDetector(frame_shape=video_provider.frame_shape, collision_time_threshold=5,
                                           n_of_subsequent_frames_required=30)
    timer = MyTimer()
    timer.start()

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        pixel_speed = speed_calculator.get_speed(depth_frame)
        timer.end_period("speed")

        time_to_collision = calculate_time_to_collision(depth_frame, pixel_speed)
        collisions = collision_detector.get_danger_regions(time_to_collision)
        timer.end_period("collision")

        depth_to_show = depth_frame / 20
        cv2.imshow("depth", depth_to_show)
        zeros = np.zeros(depth_frame.shape)
        speed_positive = pixel_speed.clip(min=0.0)
        speed_negative = pixel_speed.clip(max=0.0)
        speed_to_show = np.dstack((speed_positive, -speed_negative, zeros))
        speed_to_show /= 5.0

        # cv2.imshow("speed", speed_to_show)
        scaled_time = - time_to_collision / 5 + 1
        time_to_show = np.dstack((zeros, zeros, scaled_time))
        # cv2.imshow("time to collision", time_to_show)
        scaled_collisions_time = - collisions / 5 + 1
        collisions_to_show = np.dstack((zeros, zeros, scaled_collisions_time))
        # cv2.imshow("collisions", collisions_to_show)

        depth_to_show = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        cv2.imshow("show", np.concatenate((depth_to_show, speed_to_show, time_to_show)))

        cv2.waitKey(0)
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
