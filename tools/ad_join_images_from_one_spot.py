import os
import cv2
import numpy as np

mark_dict = {
    '0': "a)",
    '1': "b)",
    '2': "c)",
    '3': "d)",
}


def prepare_file_dict(directory):
    files = os.listdir(directory)
    files_dict = {}
    for file in files:
        filename_and_extension = os.path.splitext(file)
        if filename_and_extension[0].endswith('_comb'):
            base_name = filename_and_extension[0][:-7]
            if base_name in files_dict:
                files_dict[base_name].append(file)
            else:
                files_dict[base_name] = [file]
    return files_dict


def get_img_with_mark(directory, file):
    position = os.path.splitext(file)[0][-6]
    mark = mark_dict[position]
    img = cv2.imread(os.path.join(directory, file))
    cv2.putText(img, mark, (5, 180), cv2.FONT_HERSHEY_COMPLEX, 6, (255, 255, 0), thickness=7)
    return img


def combine_images(directory, images_names):
    images = []
    for img_name in images_names:
        images.append(get_img_with_mark(directory, img_name))
    return np.concatenate(tuple(images), axis=0)


if __name__ == '__main__':
    dir_path = "C:\\Users\\Michal\\Pictures\\magisterka\\ustalenie_pozycji"
    target_path = "C:\\Users\\Michal\\Pictures\\magisterka\\do_pracy\\ustalanie_pozycji"
    pictures_dict = prepare_file_dict(dir_path)
    for base_file_name in pictures_dict:
        image = combine_images(dir_path, pictures_dict[base_file_name])
        image_target_size = (int(image.shape[1] / 2), int(image.shape[0] / 2))
        image = cv2.resize(image, image_target_size)
        file_name = os.path.join(target_path, base_file_name + '_all' + '.png')
        print('Zapisywanie:', file_name)
        cv2.imwrite(file_name, image)


