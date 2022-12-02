"""
Druga metoda na podstawie artykułu A_single_camera_based_rear_obstacle_detection_system.pdf
"""
import math
import cv2
import sys
import numpy as np

import utilities.image_operations as img_utils
import utilities.plot_utils as plot_utils
from utilities.image_window_controller import ImageWindowController
from second_method.difference_accumulator import DifferenceAccumulator
from second_method.histogram import PolarHistogramGenerator
from second_method.histogram import HistogramAnalyser
from ipm import IPM


if __name__ == "__main__":
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\slupek\\1_obok_nikon.MOV"
    KEY_FRAME_INTERVAL = 3

    ANGULAR_APERTURE = math.radians(76)
    K_HORIZON = 3
    THETA = math.atan(math.tan(ANGULAR_APERTURE / 2) * (1 - 2.0 / K_HORIZON))
    H = 1.2
    CROPPED_FRAME_WIDTH = 720
    N_HISTOGRAM_BUCKETS = 180
    HISTOGRAM_MAX_HALF_DEG = math.radians(45)

    ipm = IPM(l=0, h=H, theta=THETA, angular_aperture=ANGULAR_APERTURE, n=CROPPED_FRAME_WIDTH)
    histogram_generator = PolarHistogramGenerator(ipm.x, ipm.y, HISTOGRAM_MAX_HALF_DEG, N_HISTOGRAM_BUCKETS)
    difference_accumulator = DifferenceAccumulator(KEY_FRAME_INTERVAL)
    histogram_analyser = HistogramAnalyser(10, math.radians(0), math.radians(30), 1.0 / math.radians(10))

    window = ImageWindowController("original")
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
    }

    video = cv2.VideoCapture(video_path)
    frame_number = 1
    success, image = video.read()
    while success:
        cropped_image = img_utils.crop_wide_frame_to_square(image)
        grey_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        difference_accumulator.add_frame(grey_cropped)
        if difference_accumulator.is_difference_accumulated():
            difference = difference_accumulator.difference_accumulator
            cv2.imshow("difference", difference)
            # transformed = ipm.transform_frame(difference, 600, 600, 100)
            # cv2.imshow("transformed", transformed)
            histogram = histogram_generator.get_histogram(difference)
            max_histogram = histogram.buckets.max()
            plot = plot_utils.get_row_plot_with_axes(histogram.buckets, mark_y=max_histogram / 3, min_val=0)

            peaks = histogram_analyser.find_obstacles(histogram)
            for peak, shade in zip(peaks, np.linspace(255, 150, len(peaks))):
                part_to_plot = np.zeros(histogram.buckets.size)
                part_to_plot[peak[0]:(peak[1] + 1)] = histogram.buckets[peak[0]:(peak[1] + 1)]
                plot_utils.append_row_plot_with_axes(plot, part_to_plot, plot_color=(0, int(shade), 0), max_val=max_histogram)
            cv2.imshow("histogram", plot)

        print("frame:", frame_number, "key:", (frame_number - 1) % KEY_FRAME_INTERVAL == 0)

        cv2.imshow("grey", grey_cropped)
        window.show_image(image)

        frame_number += 1
        success, image = video.read()


