import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utilities.timer import MyTimer
from video import VideoInterpreter
import segmentation.stixels as sxs
from monodepth2_runner import Monodepth2Runner
from utilities.image_window_controller import ImageWindowController
import utilities.image_operations as img_utils


def plot_column(depth_column):
    result_array = np.full(len(depth_column), False)
    result_array[0] = result_array[-1] = True
    sxs.calculate_stixels_ends(depth_column * depth_stixel_multiplier, result_array, stixels_threshold)

    indices = np.array(range(len(depth_column), 0, -1))
    plt.plot(depth_column, indices)
    plt.plot(depth_column[result_array], indices[result_array])
    plt.xlim([0, 30])
    plt.show()


def callback_function(x, y):
    print(x, y)
    plot_column(depth_frame[:, x])
    copy_of_frame_to_show = np.copy(frame_with_stixels)
    img_utils.paint_column_green(copy_of_frame_to_show, x)
    img_utils.paint_cross(copy_of_frame_to_show, (x, y), (0, 0, 255))
    depth_window.show_image(copy_of_frame_to_show)


if __name__ == '__main__':

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\chodnik\\3_nikon.MOV"

    video_provider = VideoInterpreter(video_path, depth_generator=Monodepth2Runner(), show_original=False)

    timer = MyTimer()
    timer.start()
    stixels_threshold = 36.0
    depth_stixel_multiplier = 20.0

    depth_window = ImageWindowController(window_name="depth", callback_function=callback_function)
    depth_window.wait_keys_dict = {
        32: depth_window.stop_waiting_for_key,
        27: sys.exit
    }
    original_frame_to_show = None

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        stixels_ends = sxs.calculate_stixels_ends_for_frame(depth_frame * depth_stixel_multiplier, stixels_threshold)
        timer.end_period("stixels")

        zeros = np.zeros(depth_frame.shape)
        depth_to_show = depth_frame / 20
        original_frame_to_show = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        # cv2.imshow("depth", original_frame_to_show)
        frame_with_stixels = np.copy(original_frame_to_show)
        frame_with_stixels[:, :, 0][stixels_ends] = 0
        frame_with_stixels[:, :, 1][stixels_ends] = 0
        # cv2.imshow("depth", frame_with_stixels)
        # wait_for_key()
        depth_window.show_image(frame_with_stixels)
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
