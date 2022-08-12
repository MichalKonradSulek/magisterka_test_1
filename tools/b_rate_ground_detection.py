import os


def get_image_pairs_from_directory(directory_path):
    files_in_dir = os.listdir(directory_path)
    qualified_files = []
    for item in files_in_dir:
        path = os.path.join(files_dir, item)
        if os.path.isfile(path):
            filename_and_extension = os.path.splitext(item)
            if filename_and_extension[1] == ".png" and not filename_and_extension[0].endswith("_true"):
                potential_true_file = os.path.join(files_dir,
                                                   filename_and_extension[0] + '_true' + filename_and_extension[1])
                if os.path.exists(potential_true_file):
                    qualified_files.append((path, potential_true_file))
    return qualified_files


if __name__ == "__main__":
    files_dir = "C:\\Users\\Michal\\Pictures\\magisterka"
    pairs_of_images = get_image_pairs_from_directory(files_dir)

    print(pairs_of_images)
