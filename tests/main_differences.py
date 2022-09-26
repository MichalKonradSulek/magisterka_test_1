import cv2
import numpy as np

from monodepth2_runner import Monodepth2Runner
from utilities.image_window_controller import ImageWindowController


if __name__ == '__main__':
    video_path = "C:\\Users\\Michal\\Videos\\magisterka\\baza_filmow\\kosz\\2_obok_nikon.MOV"

    max_depth = 20
    v_disparity_levels = 100
    curve_degree = 3
    threshold = 0.1
    tolerance = 0.7

    depth_generator = Monodepth2Runner()

    video = cv2.VideoCapture(video_path)

    success, frame = video.read()
    while success:
        resized_frame = cv2.resize(frame, (depth_generator.frame_shape[1], depth_generator.frame_shape[0]))
        depth = depth_generator.generate_depth(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)).squeeze()

        horizontal = np.abs(depth[1:, 1:] - depth[1:, :-1])
        vertical = np.abs(depth[1:, 1:] - depth[:-1, 1:])
        combined = horizontal + vertical * 2
        selected = (combined < 0.06) * 1.0
        further_than_20 = depth[1:, 1:] > 20
        selected[further_than_20] = 0.0
        # modified = cv2.erode(selected, np.ones((2, 2), np.uint8))
        modified = cv2.dilate(selected, np.ones((10, 10), np.uint8))

        cv2.imshow("original", resized_frame)
        # cv2.imshow("horizontal", horizontal * 10)
        # cv2.imshow("vertical", vertical * 10)
        # cv2.imshow("combined", combined * 5)
        cv2.imshow("selected", selected)
        cv2.imshow("modified", modified)
        cv2.imshow("depth", depth / 20)
        cv2.waitKey(0)

        success, frame = video.read()