"""
Skrypt umożliwia wybranie z filmu klatek, na których aparat jest ustawiony pod odpowiednim kątem
"""

import os
import cv2
import sys
import numpy as np

from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner
import utilities.image_operations as img_utils
import utilities.path_utils as path_utils
import utilities.plot_utils as plot_utils


column_plot = None


def save_all_frames_and_plot(position_number):
    # pozycja 0 - horyzont u góry kadru, 1 - w 1/6 kadru, 2 - w jednej trzeciej, 3 - w połowie
    image_path = path_utils.create_file_name(folder_for_images, file_name, str(position_number), ".png")
    path_plus_extension = os.path.splitext(image_path)
    depth_image_path = path_plus_extension[0] + "_depth" + path_plus_extension[1]
    plot_path = path_plus_extension[0] + "_plot" + path_plus_extension[1]
    print("saving position " + str(position_number) + ":", image_path, depth_image_path, plot_path,
          sep="\n\t", end="\n")
    cv2.imwrite(image_path, image_to_show)
    cv2.imwrite(depth_image_path, img_utils.convert_to_savable_format(depth_to_show))
    cv2.imwrite(plot_path, column_plot)


def save_pos_0():
    save_all_frames_and_plot(0)


def save_pos_1():
    save_all_frames_and_plot(1)


def save_pos_2():
    save_all_frames_and_plot(2)


def save_pos_3():
    save_all_frames_and_plot(3)


def add_important_lines_to_image(img):
    width = img.shape[1]
    height = img.shape[0]
    x_p2 = int(width / 2)
    y_p6 = int(height / 6)
    y_p3 = int(height / 3)
    y_p2 = int(height / 2)
    cv2.line(img, (0, y_p6), (width, y_p6), color=(255, 0, 0))
    cv2.line(img, (0, y_p3), (width, y_p3), color=(255, 0, 0))
    cv2.line(img, (0, y_p2), (width, y_p2), color=(255, 0, 0))
    cv2.line(img, (x_p2, 0), (x_p2, height), color=(255, 0, 0))


if __name__ == "__main__":
    folder_for_images = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\ustalenie_pozycji\\DSC_0250.MOV"
    file_name = path_utils.get_file_name_from_path(video_path)

    window = ImageWindowController()
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
        48: save_pos_0,  # 0
        49: save_pos_1,  # 1
        50: save_pos_2,  # 2
        51: save_pos_3,  # 3
    }
    depth_generator = Monodepth2Runner()

    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    while success:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(image_rgb).squeeze()

        image_to_show = np.copy(image)
        add_important_lines_to_image(image_to_show)
        depth_to_show = depth / 20
        column_plot = plot_utils.get_column_plot(depth[:, int(depth.shape[1] / 2)], min_val=0, max_val=30)

        cv2.imshow("plot", column_plot)
        cv2.imshow("depth", depth_to_show)
        window.show_image(image_to_show)

        success, image = video.read()
