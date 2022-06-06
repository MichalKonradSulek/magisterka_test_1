import numpy as np


def calculate_time_to_collision(depth_frame, speed_frame):
    return depth_frame / speed_frame.clip(min=0.0001)


class CollisionDetector:
    def __init__(self, frame_shape, collision_time_threshold, n_of_subsequent_frames_required):
        self.collision_time_threshold = collision_time_threshold
        self.counters_array = np.zeros(shape=frame_shape, dtype=np.uint8)
        self.huge_time_value = 1000
        self.n_of_subsequent_frames_required = n_of_subsequent_frames_required

    def get_danger_regions(self, time_to_collision_frame):
        elements_with_potential_danger = time_to_collision_frame < self.collision_time_threshold
        self.counters_array[elements_with_potential_danger] += 1
        self.counters_array[~elements_with_potential_danger] = 0
        array_to_return = np.ones(time_to_collision_frame.shape) * self.huge_time_value
        danger_mask = self.counters_array >= self.n_of_subsequent_frames_required
        array_to_return[danger_mask] = time_to_collision_frame[danger_mask]
        return array_to_return
