"""
Skrypt dodający osie do obrazów, zawierających wykresy.
"""

import cv2
import os

from utilities.plot_utils import add_axes


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    files = os.listdir(dir_path)
    for file in files:
        name_and_extension = os.path.splitext(file)
        if name_and_extension[0].endswith('plot'):
            plot = cv2.imread(os.path.join(dir_path, file))
            plot_with_axes = add_axes(plot, 0, 30, 0, 192, "depth [m]", "h [pix]", 85, 5, 200)
            new_path = os.path.join(dir_path, name_and_extension[0] + 'ax' + name_and_extension[1])
            cv2.imwrite(new_path, plot_with_axes)
