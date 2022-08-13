import os
import cv2
import sys

from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
from utilities.image_operations import convert_to_savable_format


def get_file_name_from_path(path):
    name_ext = os.path.basename(path)
    return os.path.splitext(name_ext)[0]


def create_file_name(destination_folder, video_name, frame_no, extension):
    new_file = video_name + "_" + str(frame_no) + extension
    return os.path.join(destination_folder, new_file)


def correct_path_if_file_exists(path):
    while os.path.exists(path):
        split_path = os.path.splitext(path)
        path = split_path[0] + 'a' + split_path[1]
    return path


def save_frame():
    path = create_file_name(folder_for_images, file_name, frame_number, ".png")
    path = correct_path_if_file_exists(path)
    print("saving:", path)
    cv2.imwrite(path, image)


def save_all_frames():
    image_path = create_file_name(folder_for_images, file_name, frame_number, ".png")
    image_path = correct_path_if_file_exists(image_path)
    path_plus_extension = os.path.splitext(image_path)
    depth_image_path = path_plus_extension[0] + "_depth" + path_plus_extension[1]
    disparity_image_path = path_plus_extension[0] + "_disp" + path_plus_extension[1]
    print("saving:", image_path, depth_image_path, disparity_image_path, sep="\n\t", end="\n")
    cv2.imwrite(image_path, image)
    cv2.imwrite(depth_image_path, convert_to_savable_format(depth_to_show))
    cv2.imwrite(disparity_image_path, convert_to_savable_format(v_disparity))


if __name__ == "__main__":
    folder_for_images = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\ustalenie_pozycji\\DSC_0250.MOV"

    depth_generator = Monodepth2Runner()
    disparity_calculator = VDisparityCalculator(depth_generator.frame_shape)
    file_name = get_file_name_from_path(video_path)
    window = ImageWindowController()
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
        115: save_frame,  # S
        97: save_all_frames,  # A
    }
    disparity_window = ImageWindowController("v_disparity", run_cv_waitkey=False, resize_factor_x_y=(2, 2))

    video = cv2.VideoCapture(video_path)
    frame_number = 1
    success, image = video.read()
    while success:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(image_rgb).squeeze()
        v_disparity = disparity_calculator.create_v_disparity(depth)
        depth_to_show = depth / 20
        cv2.imshow("depth", depth_to_show)
        disparity_window.show_image(v_disparity)
        window.show_image(image)
        frame_number += 1
        success, image = video.read()


