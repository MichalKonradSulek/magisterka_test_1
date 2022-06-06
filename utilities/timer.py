import time


class MyTimer:
    def __init__(self):
        self._last_point_in_time = None
        self.periods = []
        self.periods_names = []
        self.frame_counter = 0

    def start(self):
        self._last_point_in_time = time.time()
        self.periods.clear()
        self.periods_names.clear()

    def get_time_from_last_point(self):
        current_time = time.time()
        value_to_return = current_time - self._last_point_in_time
        self._last_point_in_time = current_time
        return value_to_return

    def end_period(self, period_name):
        self.periods.append(self.get_time_from_last_point())
        self.periods_names.append(period_name)

    def print_periods(self):
        self.frame_counter += 1
        print("frame: %4d" % self.frame_counter, end=' ')
        for name, period in zip(self.periods_names, self.periods):
            print("%s: %7.5f" % (name, period), end=' ')
        elapsed_time = sum(self.periods)
        print("elapsed: %7.5f fps: %4.1f" % (elapsed_time, 1/elapsed_time))
        self.periods.clear()
        self.periods_names.clear()




