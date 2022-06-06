import cv2
import numpy as np

from pixel_speed_analysis.collision_detection import CollisionDetector
from pixel_speed_analysis.collision_detection import calculate_time_to_collision
from pixel_speed_analysis.speed_calculator import SpeedCalculator
from video import Monodepth2VideoInterpreter

if __name__ == '__main__':

    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = Monodepth2VideoInterpreter(video_path)
    speed_calculator = SpeedCalculator(frame_shape=video_provider.frame_shape, n_of_considered_frames=30)
    collision_detector = CollisionDetector(frame_shape=video_provider.frame_shape, collision_time_threshold=5,
                                           n_of_subsequent_frames_required=30)

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        zeros = np.zeros(depth_frame.shape)

        depth_to_show = depth_frame / 20
        cv2.imshow("depth", depth_to_show)

        pixel_speed = speed_calculator.get_speed(depth_frame)
        speed_positive = pixel_speed.clip(min=0.0)
        speed_negative = pixel_speed.clip(max=0.0)
        speed_to_show = np.dstack((speed_positive, -speed_negative, zeros))
        speed_to_show /= 5.0
        cv2.imshow("speed", speed_to_show)

        time_to_collision = calculate_time_to_collision(depth_frame, pixel_speed)
        scaled_time = - time_to_collision / 5 + 1
        time_to_show = np.dstack((zeros, zeros, scaled_time))
        cv2.imshow("time to collision", time_to_show)

        collisions = collision_detector.get_danger_regions(time_to_collision)
        print(collisions.min())
        scaled_collisions_time = - collisions / 5 + 1
        collisions_to_show = np.dstack((zeros, zeros, scaled_collisions_time))
        cv2.imshow("collisions", collisions_to_show)

        cv2.waitKey(1)

        success, depth_frame = video_provider.get_next_depth_frame()
