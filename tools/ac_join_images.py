"""
Skrypt łączy trzy zdjęcia w jedno (zdjęcie x, x_depth oraz x_plotax). Wysokość zdjęcia x determinuje wysokość całości.
Wynik zapisywany jest z końcówką _comb i rozszerzeniem .png.
"""

import os
import cv2
import utilities.image_operations as img_utils


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    files = os.listdir(dir_path)
    for file in files:
        filename_and_extension = os.path.splitext(file)
        if filename_and_extension[0].endswith('_plotax'):
            original_file_name = filename_and_extension[0][:-7]
            rgb_path = os.path.join(dir_path, original_file_name + filename_and_extension[1])
            depth_path = os.path.join(dir_path, original_file_name + '_depth' + filename_and_extension[1])
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                rgb_img = cv2.imread(rgb_path)
                depth_img = cv2.imread(depth_path)
                plot = cv2.imread(os.path.join(dir_path, file))
                plot_target_width = int(rgb_img.shape[0] * plot.shape[1] / plot.shape[0])
                result = img_utils.join_pictures(
                    rgb_img, cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0])),
                    cv2.resize(plot, (plot_target_width, rgb_img.shape[0])))
                result_path = os.path.join(dir_path, original_file_name + '_comb' + '.png')
                print('Zapisywanie:', result_path)
                cv2.imwrite(result_path, result)
