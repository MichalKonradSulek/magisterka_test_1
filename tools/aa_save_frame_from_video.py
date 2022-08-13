import os
import cv2
import sys
import numpy as np

from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner
from v_disparity.v_disparity import VDisparityCalculator
import utilities.image_operations as img_utils

column_plot = None
copy_of_depth_to_show = None


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
    cv2.imwrite(depth_image_path, img_utils.convert_to_savable_format(depth_to_show))
    cv2.imwrite(disparity_image_path, img_utils.convert_to_savable_format(v_disparity))


def save_all_frames_and_plot():
    if column_plot is not None:
        image_path = create_file_name(folder_for_images, file_name, frame_number, ".png")
        image_path = correct_path_if_file_exists(image_path)
        path_plus_extension = os.path.splitext(image_path)
        depth_image_path = path_plus_extension[0] + "_depth" + path_plus_extension[1]
        disparity_image_path = path_plus_extension[0] + "_disp" + path_plus_extension[1]
        plot_path = path_plus_extension[0] + "_plot" + path_plus_extension[1]
        print("saving:", image_path, depth_image_path, disparity_image_path, plot_path,
              sep="\n\t", end="\n")
        cv2.imwrite(image_path, image)
        cv2.imwrite(depth_image_path, img_utils.convert_to_savable_format(depth_to_show))
        cv2.imwrite(disparity_image_path, img_utils.convert_to_savable_format(v_disparity))
        cv2.imwrite(plot_path, column_plot)
    else:
        print("NO PLOT TO SAVE!")


def show_column_plot(x, _):
    global column_plot
    global copy_of_depth_to_show
    column_plot = img_utils.get_column_plot(depth[:, x], min_val=0, max_val=30)
    cv2.imshow("plot", column_plot)

    copy_of_depth_to_show = np.copy(depth_to_show)
    copy_of_depth_to_show = np.dstack((copy_of_depth_to_show, copy_of_depth_to_show, copy_of_depth_to_show))
    cv2.line(copy_of_depth_to_show, (x, 0), (x, copy_of_depth_to_show.shape[0]), color=(0, 255, 0))
    depth_window.show_image(copy_of_depth_to_show)


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
        112: save_all_frames_and_plot,  # P
    }
    depth_window = ImageWindowController("depth", run_cv_waitkey=False, callback_function=show_column_plot)
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
        show_column_plot(int(depth_generator.frame_shape[1] / 2), None)
        depth_window.show_image(depth_to_show)
        disparity_window.show_image(v_disparity)
        window.show_image(image)
        frame_number += 1
        success, image = video.read()


