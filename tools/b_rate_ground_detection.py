import os
import numpy as np
import cv2


def get_image_pairs_from_directory(directory_path):
    files_in_dir = os.listdir(directory_path)
    qualified_files = []
    for item in files_in_dir:
        path = os.path.join(directory_path, item)
        if os.path.isfile(path):
            filename_and_extension = os.path.splitext(item)
            if filename_and_extension[1] == ".png" and not filename_and_extension[0].endswith("_true"):
                potential_true_file = os.path.join(directory_path,
                                                   filename_and_extension[0] + '_true' + filename_and_extension[1])
                if os.path.exists(potential_true_file):
                    qualified_files.append((path, potential_true_file))
    return qualified_files


def compare_results(from_analysis, true):
    n_true_pixels = np.count_nonzero(true)
    stamp = (true > 0)
    n_found_correctly = np.count_nonzero(from_analysis[stamp])
    n_found_incorrectly = np.count_nonzero(from_analysis[~stamp])
    return n_found_correctly/n_true_pixels, n_found_incorrectly/n_true_pixels


if __name__ == "__main__":
    # files_dir = "C:\\Users\\Michal\\Pictures\\magisterka"
    # pairs_of_images = get_image_pairs_from_directory(files_dir)

    test = cv2.imread("C:\\Users\\Michal\\Pictures\\magisterka\\misc\\test.png", cv2.IMREAD_GRAYSCALE)
    true = cv2.imread("C:\\Users\\Michal\\Pictures\\magisterka\\misc\\true.png", cv2.IMREAD_GRAYSCALE)

    found_correctly, found_incorrectly = compare_results(test, true)


    print(found_correctly, found_incorrectly)
