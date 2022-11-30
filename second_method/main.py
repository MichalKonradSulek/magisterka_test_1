"""
Druga metoda na podstawie artykułu A_single_camera_based_rear_obstacle_detection_system.pdf
"""
import math
import cv2
import sys
import numpy as np

from utilities.image_window_controller import ImageWindowController
import utilities.image_operations as img_utils
from second_method.difference_accumulator import DifferenceAccumulator
from ipm import IPM


if __name__ == "__main__":
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\slupek\\2_obok_nikon.MOV"
    KEY_FRAME_INTERVAL = 4

    angular_aperture = math.radians(76)
    k_horizon = 3
    theta = math.atan(math.tan(angular_aperture / 2) * (1 - 2.0 / k_horizon))
    print(math.degrees(theta))
    h = 1.2
    n = 720
    ipm = IPM(l=0, h=1.2, theta=theta, angular_aperture=angular_aperture, n=n)

    difference_accumulator = DifferenceAccumulator(KEY_FRAME_INTERVAL)
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
        # if difference_accumulator.is_difference_accumulated():
        #     cv2.imshow("difference", difference_accumulator.difference_accumulator)
        transformed = ipm.transform_frame(grey_cropped, 600, 600, 100)

        print("frame:", frame_number, "key:", (frame_number - 1) % KEY_FRAME_INTERVAL == 0)

        cv2.imshow("transformed", transformed)
        cv2.imshow("grey", grey_cropped)
        window.show_image(image)

        frame_number += 1
        success, image = video.read()


