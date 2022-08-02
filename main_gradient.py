import sys

import cv2
import numpy as np
import queue

from utilities.timer import MyTimer
from video import Monodepth2VideoInterpreter
from monodepth2_runner import Monodepth2Runner


def check_pixel(x, y, included_pixels, depth, pix_queue, reference_depth, max_grad):
    if not included_pixels[y, x] and reference_depth - max_grad <= depth[y, x] <= reference_depth + max_grad:
        included_pixels[y, x] = True
        pix_queue.put((y, x))


def expand_around_pixel(depth, included_pixels, pix_coordinates, grad_x, grad_y, pix_queue):
    y = pix_coordinates[0]
    x = pix_coordinates[1]
    if x > 0:
        check_pixel(x - 1, y, included_pixels, depth, pix_queue, depth[y, x], grad_x)
    if x < depth.shape[1] - 1:
        check_pixel(x + 1, y, included_pixels, depth, pix_queue, depth[y, x], grad_x)
    if y > 0:
        check_pixel(x, y - 1, included_pixels, depth, pix_queue, depth[y, x], grad_y)
    if y < depth.shape[0] - 1:
        check_pixel(x, y + 1, included_pixels, depth, pix_queue, depth[y, x], grad_y)
    # included_neighbours = depth[x, y] - max_grad <= depth[x - 1:x + 1, y - 1:y + 1] <= depth[x, y] + max_grad
    # included_pixels[x - 1:x + 1, y - 1:y + 1] = included_pixels[x-1:x+1, y-1:y+1] or included_neighbours


def find_object(depth, grad_x, grad_y, start_point_y_x):
    included_pixels = np.full(depth.shape, False)
    included_pixels[start_point_y_x] = True
    pix_queue = queue.Queue()
    pix_queue.put(start_point_y_x)
    while not pix_queue.empty():
        expand_around_pixel(depth, included_pixels, pix_queue.get(), grad_x, grad_y, pix_queue)
    return included_pixels


def paint_cross(mat, center, color):
    x = center[0]
    y = center[1]
    cv2.line(mat, (x - 2, y), (x + 2, y), color)
    cv2.line(mat, (x, y - 2), (x, y + 2), color)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        shape = find_object(depth_frame, max_grad_x, max_grad_y, (y, x))
        print(np.count_nonzero(shape))
        copy_of_frame_to_show = np.copy(original_frame_to_show)
        copy_of_frame_to_show[:, :, 0][shape] = 0
        copy_of_frame_to_show[:, :, 1][shape] = 0
        paint_cross(copy_of_frame_to_show, (x, y), (0, 0, 255))
        cv2.imshow("depth", copy_of_frame_to_show)


def wait_for_key():
    is_still_waiting = True
    while is_still_waiting:
        key_pressed = cv2.waitKey(100)
        if key_pressed & 0xFF == 32:
            is_still_waiting = False
        elif key_pressed & 0xFF == 27:
            sys.exit()


if __name__ == '__main__':

    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142748920.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142911005.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_142953829.mp4"
    video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143053656.mp4"
    # video_path = "C:\\Users\\Michal\\Videos\\VID_20220517_143324266.mp4"

    video_provider = Monodepth2VideoInterpreter(video_path, depth_generator=Monodepth2Runner(), show_original=False)

    timer = MyTimer()
    timer.start()

    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", mouse_callback)
    original_frame_to_show = None
    max_grad_x = 0.01
    max_grad_y = 0.05

    success, depth_frame = video_provider.get_next_depth_frame()
    while success:
        timer.end_period("depth")

        zeros = np.zeros(depth_frame.shape)
        depth_to_show = depth_frame / 20
        original_frame_to_show = np.dstack((depth_to_show, depth_to_show, depth_to_show))
        cv2.imshow("depth", original_frame_to_show)
        wait_for_key()
        timer.end_period("show")

        timer.print_periods()

        success, depth_frame = video_provider.get_next_depth_frame()
