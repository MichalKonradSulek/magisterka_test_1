import numpy as np


class DifferenceAccumulator:
    """
    Akumulator zbiera różnice między kluczową klatką a kolejnymi klatkami. Różnica jest gotowa, gdy zostały zebrane
    różnice dla wszystkich klatek (zaraz przed następną kluczową klatką)
    """
    def __init__(self, key_frame_interval):
        self._key_frame_interval = key_frame_interval
        self._frame_counter = 0
        self._key_frame = None
        self.difference_accumulator = None
        self._next_key_frame = None

    def add_frame(self, frame):
        if 0 < self._frame_counter < self._key_frame_interval:
            self._accumulate_difference(frame)
            self._frame_counter += 1
        elif self._frame_counter == self._key_frame_interval or self._frame_counter == 0:
            self._key_frame = frame
            self.difference_accumulator = np.zeros(frame.shape, dtype=frame.dtype)
            self._frame_counter = 1
        else:
            raise Exception("Bad frame counter")

    def is_difference_accumulated(self):
        return self._frame_counter == self._key_frame_interval

    def _accumulate_difference(self, frame):
        difference = self._key_frame.astype("int16") - frame
        difference = np.absolute(difference.clip(0, 255).astype("uint8"))
        self.difference_accumulator += difference
