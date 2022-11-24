"""
Skrypt umożliwia zachowanie wybranej klatki z filmu
"""

import os
import cv2
import sys
import numpy as np

from u_disparity.u_disparity import UDisparityCalculator
from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
import utilities.image_operations as img_utils
import utilities.path_utils as path_utils
import utilities.plot_utils as plot_utils

column_plot = None
copy_of_depth_to_show = None


def correct_path_if_file_exists(path):
    while os.path.exists(path):
        split_path = os.path.splitext(path)
        path = split_path[0] + 'a' + split_path[1]
    return path


def save_frame():
    path = path_utils.create_file_name(folder_for_images, file_name, str(frame_number), ".png")
    path = correct_path_if_file_exists(path)
    print("saving:", path)
    cv2.imwrite(path, image)


def save_all_frames():
    # FIXME dodaj zapisywanie u-disparity
    image_path = path_utils.create_file_name(folder_for_images, file_name, str(frame_number), ".png")
    image_path = correct_path_if_file_exists(image_path)
    path_plus_extension = os.path.splitext(image_path)
    depth_image_path = path_plus_extension[0] + "_depth" + path_plus_extension[1]
    disparity_image_path = path_plus_extension[0] + "_disp" + path_plus_extension[1]
    print("saving:", image_path, depth_image_path, disparity_image_path, sep="\n\t", end="\n")
    cv2.imwrite(image_path, image)
    cv2.imwrite(depth_image_path, img_utils.convert_to_savable_format(depth_to_show))
    cv2.imwrite(disparity_image_path, img_utils.convert_to_savable_format(v_disparity))


def get_object_mask(depth_map, contour):
    x, y, w, h = cv2.boundingRect(contour)
    min_x = x
    max_x = x + w
    min_d = (max_depth / disparity_levels) * (y + 0.5)
    max_d = (max_depth / disparity_levels) * (y + h + 0.5)
    object_mask = np.logical_and(depth_map >= min_d, depth_map <= max_d)
    object_mask[:, 0:min_x] = False
    object_mask[:, (max_x + 1):] = False
    return object_mask


if __name__ == "__main__":
    folder_for_images = "C:\\Users\\Michal\\Pictures\\magisterka\\u_disparity"
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\slupek\\3_obok_nikon.MOV"
    max_depth = 20
    disparity_levels = 100

    depth_generator = Monodepth2Runner()
    u_disparity_calculator = UDisparityCalculator(depth_generator.frame_shape, disparity_levels, max_depth)
    v_disparity_calculator = VDisparityCalculator(depth_generator.frame_shape, disparity_levels, max_depth)
    u_disparity_window = ImageWindowController("u_disparity", resize_factor_x_y=(1, 2), run_cv_waitkey=False)
    v_disparity_window = ImageWindowController("v_disparity", resize_factor_x_y=(2, 2), run_cv_waitkey=False)
    file_name = path_utils.get_file_name_from_path(video_path)
    window = ImageWindowController()
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
        115: save_frame,  # S
        97: save_all_frames,  # A
    }

    video = cv2.VideoCapture(video_path)
    frame_number = 1
    success, image = video.read()
    while success:
        resized_image = cv2.resize(image, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        depth = depth_generator.generate_depth(image_rgb).squeeze()
        v_disparity = v_disparity_calculator.create_v_disparity(depth)
        u_disparity = u_disparity_calculator.create_u_disparity(depth)
        depth_to_show = depth / 20
        # cv2.imshow("depth", depth_to_show)
        u_disparity[u_disparity < 0.05] = 0.0
        u_disparity[u_disparity >= 0.05] = 1.0

        u_disparity_8bit = (u_disparity * 255).astype('uint8')
        img, contours, hierarchy = cv2.findContours(u_disparity_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_th = 0  # FIXME to prawdopodobnie wymaga zmiany
        contour_th = 5

        filtered_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > area_th and cv2.arcLength(cnt, True) > contour_th:
                filtered_contours.append(cnt)

        depth_bgr = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        u_disparity_8bit_bgr = np.dstack((u_disparity_8bit, u_disparity_8bit, u_disparity_8bit))
        objects_colors = [i * 1.0 / len(filtered_contours) for i in range(len(filtered_contours))]

        for cnt, object_color in zip(filtered_contours, objects_colors):
            mask = get_object_mask(depth, cnt)
            depth_bgr[:, :, 0][mask] = 1.0
            depth_bgr[:, :, 1][mask] = object_color
            depth_bgr[:, :, 2][mask] = 0.0

        cv2.imshow("object", depth_bgr)

        cv2.drawContours(u_disparity_8bit_bgr, filtered_contours, -1, (0, 255, 0), -1)
        cv2.imshow("contours", u_disparity_8bit_bgr)



        v_disparity_window.show_image(v_disparity * 4.0)
        u_disparity_window.show_image(u_disparity * 4.0)
        window.show_image(resized_image)
        frame_number += 1
        success, image = video.read()


