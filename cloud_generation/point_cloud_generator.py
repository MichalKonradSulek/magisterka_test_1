import numpy as np


class PointCloudGenerator:
    def __init__(self, width_pix, height_pix, sensor_width, sensor_height, focal_length):
        self.width_pix = width_pix
        self.height_pix = height_pix
        self.pix_size_width = sensor_width / width_pix
        self.pix_size_height = sensor_height / height_pix
        self.focal_length = focal_length

    def generate(self, depth_array):
        result = []
        for ih, iw in np.ndindex(depth_array.shape):
            depth = depth_array[ih, iw]
            x = (depth + self.focal_length) * (self.width_pix / 2.0 - iw - 1) * self.pix_size_width / self.focal_length
            y = (depth + self.focal_length) * (self.height_pix / 2.0 - ih - 1) * self.pix_size_height / self.focal_length
            result.append([-x, -y, depth])
        return np.array(result)
