"""
Skrypt dodający osie do obrazów, zawierających wykresy.
"""

import cv2
import numpy as np
import os


def detect_bg_color(img):
    n_of_pixels = img.shape[0] * img.shape[1]
    bg_b = np.argmax(np.bincount(np.reshape(img[:, :, 0], n_of_pixels)))
    bg_g = np.argmax(np.bincount(np.reshape(img[:, :, 1], n_of_pixels)))
    bg_r = np.argmax(np.bincount(np.reshape(img[:, :, 2], n_of_pixels)))
    return bg_b, bg_g, bg_r


def extend_image(img, color, left, down, up=0, right=0):
    extended_layer = np.ones((img.shape[0] + down + up, img.shape[1] + left + right))
    extended_img = np.dstack((extended_layer * color[0], extended_layer * color[1], extended_layer * color[2]))
    extended_img[up:(img.shape[0] + up), left:(img.shape[1] + left), :] = img
    return extended_img


def text_width(text):
    return len(text) * 10


def add_axes(plot_img, x_min_val, x_max_val, y_min_val, y_max_val, x_label, y_label, x_label_length, mark_x, mark_y):
    th_left = 20
    th_down = 20
    mark_length = 10
    axis_color = (0, 0, 0)

    bg_color = detect_bg_color(plot_img)
    extended_plot = extend_image(plot_img, bg_color, th_left, th_down)

    height = extended_plot.shape[0]
    width = extended_plot.shape[1]

    cv2.line(extended_plot, (0, height - th_down), (width, height - th_down), axis_color)
    cv2.line(extended_plot, (th_left, 0), (th_left, height), axis_color)
    for w in range(mark_x, x_max_val, mark_x):
        w_pix = int(plot_img.shape[1] * (w - x_min_val) / (x_max_val - x_min_val) + th_left)
        cv2.line(extended_plot, (w_pix, height - th_down), (w_pix, height - th_down + mark_length), axis_color)
        cv2.putText(extended_plot, str(w), (w_pix + 2, height - th_down + 15), cv2.FONT_HERSHEY_PLAIN, 1, axis_color)
    cv2.putText(extended_plot, x_label, (width - x_label_length, height - th_down + 15),
                cv2.FONT_HERSHEY_PLAIN, 1, axis_color)
    for h in range(mark_y, y_max_val, mark_y):
        h_pix = height - th_down - int(plot_img.shape[0] * (h - y_min_val) / (y_max_val - y_min_val))
        cv2.line(extended_plot, (th_left - mark_length, h_pix), (th_left, h_pix), axis_color)
        cv2.putText(extended_plot, str(h), (th_left - text_width(str(h)) - 2, h_pix - 2), cv2.FONT_HERSHEY_PLAIN, 1,
                    axis_color)
    cv2.putText(extended_plot, y_label, (th_left + 2, 0 + 15),
                cv2.FONT_HERSHEY_PLAIN, 1, axis_color)
    return extended_plot


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    files = os.listdir(dir_path)
    for file in files:
        name_and_extension = os.path.splitext(file)
        if name_and_extension[0].endswith('plot'):
            plot = cv2.imread(os.path.join(dir_path, file))
            plot_with_axes = add_axes(plot, 0, 20, 0, 192, "depth [m]", "h [pix]", 85, 5, 200)
            new_path = os.path.join(dir_path, name_and_extension[0] + 'ax' + name_and_extension[1])
            cv2.imwrite(new_path, plot_with_axes)
