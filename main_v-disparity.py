import cv2

from utilities.timer import MyTimer
from v_disparity.v_disparity import VDisparityCalculator
from video import Monodepth2VideoInterpreter

if __name__ == '__main__':

    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = Monodepth2VideoInterpreter(video_path)
    disparity_calculator = VDisparityCalculator(video_provider.frame_shape)
    timer = MyTimer()
    timer.start()

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        v_disparity = disparity_calculator.create_v_disparity(depth_frame)
        v_disparity = v_disparity > 0.5
        timer.end_period("v-disparity")
        print(v_disparity.shape)

        depth_to_show = depth_frame / 20
        cv2.imshow("depth", depth_to_show)
        v_disparity_to_show_size = (v_disparity.shape[1] * 2, v_disparity.shape[0] * 2)
        cv2.imshow("v-disparity", cv2.resize(v_disparity, v_disparity_to_show_size))
        cv2.waitKey(1)
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
