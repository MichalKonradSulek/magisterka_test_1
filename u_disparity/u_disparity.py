import numpy as np
import cv2


class UDisparityCalculator:
    def __init__(self, frame_shape, levels_of_depth=100, max_depth=100.):
        self.frame_shape = frame_shape
        self.levels_of_depth = levels_of_depth
        self.max_depth = max_depth

    def create_u_disparity(self, depth_frame):
        result = np.zeros((self.levels_of_depth, self.frame_shape[1]))
        for i in range(self.frame_shape[1]):
            result[:, i] = cv2.calcHist(images=[depth_frame[:, i]], channels=[0], mask=None,
                                        histSize=[self.levels_of_depth], ranges=[0, self.max_depth]).flatten()
        return result / self.frame_shape[0]
