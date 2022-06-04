import numpy as np


class SpeedCalculator:
    def __init__(self, input_video_fps=30):
        self.previous_frame = None
        self.frame_duration = 1.0 / input_video_fps

    def get_pixel_speed(self, new_frame):
        if self.previous_frame is None:
            self.previous_frame = new_frame
            return np.zeros(new_frame.shape)
        else:
            value_to_return = (new_frame - self.previous_frame) / self.frame_duration
            self.previous_frame = new_frame
            return value_to_return
