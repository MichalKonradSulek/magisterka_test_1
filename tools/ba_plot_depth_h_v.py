"""
Skrypt umożliwia zachowanie wybranej klatki z filmu
"""

import os
import cv2
import sys
import numpy as np

from utilities.image_window_controller import ImageWindowController
from monodepth2_runner import Monodepth2Runner
import utilities.image_operations as img_utils
import utilities.plot_utils as plot_utils

selected_x = None
selected_y = None
plot = None
mode = 'v'


def save_plot():
    if mode == 'h' and selected_y is not None:
        filepath = os.path.splitext(image_path)[0] + '_ploth_' + str(selected_y) + '.png'
    elif mode == 'v' and selected_x is not None:
        filepath = os.path.splitext(image_path)[0] + '_plotv_' + str(selected_x) + '.png'
    else:
        assert False
    print('Saving:', filepath)
    cv2.imwrite(filepath, plot)


def show_h_plot(y):
    global plot
    y_on_original = int(y * image.shape[0] / depth.shape[0])
    copy_of_image = np.copy(image)
    cv2.line(copy_of_image, (0, y_on_original), (image.shape[1], y_on_original), (255, 0, 0), thickness=2)
    plot = plot_utils.get_row_plot_with_axes(depth[y, :], y_label="depth [m]", min_val=0, max_val=20, mark_y=5)
    cv2.imshow("plot", plot)
    window.show_image(copy_of_image)


def show_v_plot(x):
    global plot
    x_on_original = int(x * image.shape[1] / depth.shape[1])
    copy_of_image = np.copy(image)
    cv2.line(copy_of_image, (x_on_original, 0), (x_on_original, image.shape[0]), (255, 0, 0), thickness=2)
    plot = plot_utils.get_column_plot_with_axes(depth[:, x], x_label="depth [m]", x_label_length=85,
                                                min_val=0, max_val=20, mark_x=5)
    cv2.imshow("plot", plot)
    window.show_image(copy_of_image)


def select_point_and_plot(x, y):
    global selected_x
    global selected_y
    print('Selected: (', x, ', ', y, ')', sep="")
    selected_x = x
    selected_y = y
    copy_of_depth_to_show = np.dstack((np.copy(depth), np.copy(depth), np.copy(depth)))
    img_utils.paint_cross(copy_of_depth_to_show, (x, y), (0, 0, 255))
    depth_window.show_image(copy_of_depth_to_show / 20, do_not_run_waitkey=True)
    if mode == 'v':
        show_v_plot(x)
    elif mode == 'h':
        show_h_plot(y)


def change_mode_to_v():
    global mode
    mode = 'v'
    if selected_x is not None:
        show_v_plot(selected_x)


def change_mode_to_h():
    global mode
    mode = 'h'
    if selected_y is not None:
        show_h_plot(selected_y)


if __name__ == "__main__":
    image_path = "C:\\Users\\Michal\\Pictures\\magisterka\\pusty_chodnik\\1_nikon_45.png"

    depth_generator = Monodepth2Runner()
    window = ImageWindowController()
    window.wait_keys_dict = {
        32: window.stop_waiting_for_key,  # space bar
        27: sys.exit,  # esc
        # 112: save_h_plot,  # P
        # 105: save_v_plot,  # I
        115: save_plot,  # S
        104: change_mode_to_h,  # H
        118: change_mode_to_v,  # V
    }
    depth_window = ImageWindowController("depth", run_cv_waitkey=False, callback_function=select_point_and_plot)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
    depth = depth_generator.generate_depth(image_rgb).squeeze()
    depth_to_show = depth / 20
    depth_window.show_image(depth_to_show)
    window.show_image(image)
