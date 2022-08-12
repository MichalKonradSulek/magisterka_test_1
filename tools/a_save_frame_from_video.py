import os
import cv2
import sys

from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner


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


if __name__ == "__main__":
    folder_for_images = "C:\\Users\\Michal\\Pictures\\magisterka"
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\zywoplot\\obok_1_nikon.MOV"

    depth_generator = Monodepth2Runner()
    file_name = get_file_name_from_path(video_path)
    window = ImageWindowController()
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
        115: save_frame,  # S
    }

    video = cv2.VideoCapture(video_path)
    frame_number = 1
    success, image = video.read()
    while success:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(image_rgb).squeeze()
        cv2.imshow("depth", depth / 20)
        window.show_image(image)
        frame_number += 1
        success, image = video.read()


