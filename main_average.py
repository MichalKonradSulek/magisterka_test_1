import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from utilities.timer import MyTimer
from video import VideoInterpreter
from pixel_speed_analysis.depth_equalization import DepthEqualizer
from monodepth2_runner import Monodepth2Runner

import pixel_speed_analysis.speed_calculator

if __name__ == '__main__':

    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = VideoInterpreter(video_path, depth_generator=Monodepth2Runner(), show_original=False)
    depth_equalizer = DepthEqualizer(n_of_considered_frames=20)

    timer = MyTimer()
    timer.start()

    i = 0
    plot_x = []
    plot_y = []
    plot_reg = []

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        equalized_depth = depth_equalizer.get_equalized_depth(depth_frame)

        i += 1
        current_mean = depth_frame.mean()

        plot_x.append(i)
        plot_y.append(current_mean)
        plot_reg.append(depth_equalizer._get_equalized_mean(current_mean))
        if i % 10 == 0:
            plt.plot(plot_x, plot_y)
            plt.plot(plot_x, plot_reg)
            plt.show()
        timer.end_period("plot")

        zeros = np.zeros(depth_frame.shape)
        depth_to_show = depth_frame / 20
        original_frame_to_show = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        cv2.imshow("depth", original_frame_to_show)

        equalized_depth_to_show = equalized_depth / 20
        cv2.imshow("equalized_depth", equalized_depth_to_show)

        cv2.waitKey(0)
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
