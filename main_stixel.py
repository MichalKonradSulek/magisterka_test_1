import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utilities.timer import MyTimer
from video import VideoInterpreter
import segmentation.stixels as sxs
from monodepth2_runner import Monodepth2Runner


def paint_cross(mat, center, color):
    x = center[0]
    y = center[1]
    cv2.line(mat, (x - 2, y), (x + 2, y), color)
    cv2.line(mat, (x, y - 2), (x, y + 2), color)


def paint_column(mat, x):
    mat[:, x, 0] = 0.0
    mat[:, x, 2] = 0.0


def plot_column(depth_column):
    result_array = np.full(len(depth_column), False)
    result_array[0] = result_array[-1] = True
    sxs.calculate_stixels_ends(depth_column * depth_stixel_multiplier, result_array, stixels_threshold)

    indices = np.array(range(len(depth_column), 0, -1))
    plt.plot(depth_column, indices)
    plt.plot(depth_column[result_array], indices[result_array])
    plt.xlim([0, 30])
    plt.show()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        plot_column(depth_frame[:, x])
        copy_of_frame_to_show = np.copy(frame_with_stixels)
        paint_column(copy_of_frame_to_show, x)
        paint_cross(copy_of_frame_to_show, (x, y), (0, 0, 255))
        cv2.imshow("depth", copy_of_frame_to_show)


def wait_for_key():
    is_still_waiting = True
    while is_still_waiting:
        key_pressed = cv2.waitKey(100)
        if key_pressed & 0xFF == 32:
            is_still_waiting = False
        elif key_pressed & 0xFF == 27:
            sys.exit()


if __name__ == '__main__':

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = VideoInterpreter(video_path, depth_generator=Monodepth2Runner(), show_original=False)

    timer = MyTimer()
    timer.start()
    stixels_threshold = 36.0
    depth_stixel_multiplier = 20.0

    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", mouse_callback)
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
        cv2.imshow("depth", frame_with_stixels)
        wait_for_key()
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
