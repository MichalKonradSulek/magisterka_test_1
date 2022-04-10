import time


class MyTimer:
    def __init__(self):
        self._last_point_in_time = None

    def start(self):
        self._last_point_in_time = time.time()

    def get_time_from_last_point(self):
        current_time = time.time()
        value_to_return = current_time - self._last_point_in_time
        self._last_point_in_time = current_time
        return value_to_return
