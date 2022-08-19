import cv2
import numpy as np


def convert_to_savable_format(image):
    return (image * 255).clip(0, 255).astype('uint8')


def paint_cross(mat, center_x_y, color):
    x = center_x_y[0]
    y = center_x_y[1]
    cv2.line(mat, (x - 2, y), (x + 2, y), color)
    cv2.line(mat, (x, y - 2), (x, y + 2), color)


def paint_column_green(mat, x):
    mat[:, x, 0] = 0.0
    mat[:, x, 2] = 0.0


def _scale(min_val, max_val, target_max_val, val):
    return (val - min_val) * target_max_val / (max_val - min_val)


def get_column_plot(depth_column, min_val=None, max_val=None, bg_color=(255, 255, 255), plot_color=(255, 0, 0),
                    plot_size_x_y=(800, 600), line_thickness=1):
    if min_val is None:
        min_val = depth_column.min()
    if max_val is None:
        max_val = depth_column.max()
    bg_b = np.ones([plot_size_x_y[1], plot_size_x_y[0]], dtype='uint8') * bg_color[0]
    bg_g = np.ones([plot_size_x_y[1], plot_size_x_y[0]], dtype='uint8') * bg_color[1]
    bg_r = np.ones([plot_size_x_y[1], plot_size_x_y[0]], dtype='uint8') * bg_color[2]
    bg = np.dstack((bg_b, bg_g, bg_r))
    for i in range(len(depth_column) - 1):
        x1 = int(_scale(min_val, max_val, plot_size_x_y[0], depth_column[i]))
        y1 = int(_scale(0, len(depth_column) - 1, plot_size_x_y[1], i))
        x2 = int(_scale(min_val, max_val, plot_size_x_y[0], depth_column[i + 1]))
        y2 = int(_scale(0, len(depth_column) - 1, plot_size_x_y[1], i + 1))
        cv2.line(bg, (x1, y1), (x2, y2), color=plot_color, thickness=line_thickness)
    return bg


def join_pictures(img1, img2, img3):
    return np.concatenate((img1, img2, img3), axis=1)
