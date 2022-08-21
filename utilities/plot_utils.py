import numpy as np
import cv2


def _scale(min_val, max_val, target_max_val, val):
    return (val - min_val) * target_max_val / (max_val - min_val)


def _prepare_background(size_x_y, color):
    bg_b = np.ones([size_x_y[1], size_x_y[0]], dtype='uint8') * color[0]
    bg_g = np.ones([size_x_y[1], size_x_y[0]], dtype='uint8') * color[1]
    bg_r = np.ones([size_x_y[1], size_x_y[0]], dtype='uint8') * color[2]
    return np.dstack((bg_b, bg_g, bg_r))


def get_column_plot(depth_column, min_val=None, max_val=None, bg_color=(255, 255, 255), plot_color=(255, 0, 0),
                    plot_size_x_y=(800, 600), line_thickness=1):
    if min_val is None:
        min_val = depth_column.min()
    if max_val is None:
        max_val = depth_column.max()
    bg = _prepare_background(plot_size_x_y, bg_color)
    for i in range(len(depth_column) - 1):
        x1 = int(_scale(min_val, max_val, plot_size_x_y[0], depth_column[i]))
        y1 = int(_scale(0, len(depth_column) - 1, plot_size_x_y[1], i))
        x2 = int(_scale(min_val, max_val, plot_size_x_y[0], depth_column[i + 1]))
        y2 = int(_scale(0, len(depth_column) - 1, plot_size_x_y[1], i + 1))
        cv2.line(bg, (x1, y1), (x2, y2), color=plot_color, thickness=line_thickness)
    return bg


def get_row_plot(depth_row, min_val=None, max_val=None, bg_color=(255, 255, 255), plot_color=(255, 0, 0),
                 plot_size_x_y=(800, 600), line_thickness=1):
    if min_val is None:
        min_val = depth_row.min()
    if max_val is None:
        max_val = depth_row.max()
    bg = _prepare_background(plot_size_x_y, bg_color)
    for i in range(len(depth_row) - 1):
        x1 = int(_scale(0, len(depth_row) - 1, plot_size_x_y[0], i))
        y1 = plot_size_x_y[1] - int(_scale(min_val, max_val, plot_size_x_y[1], depth_row[i]))
        x2 = int(_scale(0, len(depth_row) - 1, plot_size_x_y[0], i + 1))
        y2 = plot_size_x_y[1] - int(_scale(min_val, max_val, plot_size_x_y[1], depth_row[i + 1]))
        cv2.line(bg, (x1, y1), (x2, y2), color=plot_color, thickness=line_thickness)
    return bg


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


def get_column_plot_with_axes(depth_column, x_label, x_label_length, min_val=None, max_val=None, bg_color=(255, 255, 255),
                              plot_color=(255, 0, 0), plot_size_x_y=(800, 600), line_thickness=1, mark_x=10):
    if min_val is None:
        min_val = depth_column.min()
    if max_val is None:
        max_val = depth_column.max()
    plot = get_column_plot(depth_column, min_val=min_val, max_val=max_val, bg_color=bg_color, plot_color=plot_color,
                           plot_size_x_y=plot_size_x_y, line_thickness=line_thickness)
    return add_axes(plot, x_min_val=min_val, x_max_val=max_val, y_min_val=0, y_max_val=len(depth_column),
                    x_label=x_label, y_label="h [pix]", x_label_length=x_label_length, mark_x=mark_x,
                    mark_y=2*len(depth_column))


def get_row_plot_with_axes(depth_row, y_label, min_val=None, max_val=None, bg_color=(255, 255, 255),
                           plot_color=(255, 0, 0), plot_size_x_y=(800, 600), line_thickness=1, mark_y=10):
    if min_val is None:
        min_val = depth_row.min()
    if max_val is None:
        max_val = depth_row.max()
    plot = get_row_plot(depth_row, min_val=min_val, max_val=max_val, bg_color=bg_color, plot_color=plot_color,
                        plot_size_x_y=plot_size_x_y, line_thickness=line_thickness)
    return add_axes(plot, x_min_val=0, x_max_val=len(depth_row), y_min_val=min_val, y_max_val=max_val,
                    x_label="w [pix]", y_label=y_label, x_label_length=65, mark_x=2*len(depth_row),
                    mark_y=5)
