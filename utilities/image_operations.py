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


def join_pictures(img1, img2, img3):
    return np.concatenate((img1, img2, img3), axis=1)
