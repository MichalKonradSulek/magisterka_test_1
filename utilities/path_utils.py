import os


def get_file_name_from_path(path):
    name_ext = os.path.basename(path)
    return os.path.splitext(name_ext)[0]


def create_file_name(destination_folder, base_name, suffix, extension):
    new_file = base_name + "_" + suffix + extension
    return os.path.join(destination_folder, new_file)
