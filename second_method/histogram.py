import numpy as np


class Histogram:
    def __init__(self, buckets, max_half_angle):
        self.buckets = buckets
        self.max_half_angle = max_half_angle
        self.bucket_width = 2 * max_half_angle / buckets.size


class PolarHistogramGenerator:
    def __init__(self, x_transformation, y_transformation, max_half_angle, n_of_buckets):
        self._n_of_buckets = n_of_buckets
        self._divisions = np.linspace(-max_half_angle, max_half_angle, n_of_buckets + 1)
        self._create_assignment_array(x_transformation, y_transformation)
        self.max_half_angle = max_half_angle

    def _create_assignment_array(self, x_array, y_array):
        temp = y_array / x_array
        angles = np.arctan(temp)
        # -1 is not assigned to any bucket
        self._assignment_array = -1 * np.ones(x_array.shape, dtype="int")
        for i in range(self._n_of_buckets):
            selected_cells = np.logical_and(angles >= self._divisions[i], angles < self._divisions[i + 1])
            self._assignment_array[selected_cells] = i
        # remove points over horizon
        points_over_horizon = x_array < 0
        self._assignment_array[points_over_horizon] = -1

    def get_histogram(self, frame):
        histogram = np.zeros(self._n_of_buckets)
        for i in range(self._n_of_buckets):
            selected_cells = self._assignment_array == i
            histogram[i] = (frame[selected_cells]).sum()
        return Histogram(histogram, self.max_half_angle)


class HistogramAnalyser:
    def __init__(self, n_of_slices, min_width, max_width, amp_width_ratio_threshold):
        self._n_of_slices = n_of_slices
        self._min_width = min_width
        self._max_width = max_width
        self._amp_width_ratio_threshold = amp_width_ratio_threshold

    def find_obstacles(self, histogram):
        max_amplitude = histogram.buckets.max()
        scaled_buckets = histogram.buckets / max_amplitude
        considered_peaks = set()
        for i in np.linspace(0.0, 1.0, num=self._n_of_slices, endpoint=False):
            potential_peaks_ranges = self._get_potential_peaks(scaled_buckets, i)
            potential_peaks_ranges = self._filter_by_thresholds(potential_peaks_ranges, histogram.bucket_width)
            potential_peaks = self._get_peaks_indices(potential_peaks_ranges, scaled_buckets)
            considered_peaks.update(potential_peaks)
        peaks = []
        for potential_peak in considered_peaks:
            peak = self._check_if_peak_valid(potential_peak, scaled_buckets, histogram.bucket_width)
            if peak is not None:
                peaks.append(peak)
        return peaks

    def _get_potential_peaks(self, scaled_buckets, slice_value):
        potential_peaks = []
        peak_start = None
        if scaled_buckets[0] >= slice_value:
            peak_start = 0
        for i in range(1, len(scaled_buckets)):
            if peak_start is None:
                if scaled_buckets[i] >= slice_value:
                    peak_start = i
            else:
                if scaled_buckets[i] < slice_value:
                    potential_peaks.append((peak_start, i))
                    peak_start = None
        if peak_start is not None:
            potential_peaks.append((peak_start, len(scaled_buckets)))
        return potential_peaks

    def _filter_by_thresholds(self, potential_peaks, bucket_width):
        filtered = []
        for (begin, end) in potential_peaks:
            if self._min_width <= (end - begin) * bucket_width <= self._max_width:
                filtered.append((begin, end))
        return filtered

    def _get_peaks_indices(self, ranges, scaled_buckets):
        peaks = []
        for (begin, end) in ranges:
            peak_index = (scaled_buckets[begin:end]).argmax() + begin
            peaks.append(peak_index)
        return peaks

    def _check_if_peak_valid(self, potential_peak, scaled_buckets, bucket_width):
        amplitude = scaled_buckets[potential_peak]
        half_amplitude = amplitude / 2
        left_margin = self._get_left_margin(potential_peak, scaled_buckets, half_amplitude)
        right_margin = self._get_right_margin(potential_peak, scaled_buckets, half_amplitude)
        width = (right_margin - left_margin) * bucket_width
        if amplitude / width > self._amp_width_ratio_threshold:
            return left_margin, right_margin
        else:
            return None

    def _get_left_margin(self, peak_index, scaled_buckets, half_amplitude):
        left_margin = None
        for i in range(peak_index - 1, 0, -1):
            if scaled_buckets[i] < half_amplitude:
                left_margin = i + 1
                break
        if left_margin is None:
            left_margin = 0
        return left_margin

    def _get_right_margin(self, peak_index, scaled_buckets, half_amplitude):
        right_margin = None
        for i in range(peak_index + 1, len(scaled_buckets) - 1):
            if scaled_buckets[i] < half_amplitude:
                right_margin = i
                break
        if right_margin is None:
            right_margin = len(scaled_buckets)
        return right_margin
