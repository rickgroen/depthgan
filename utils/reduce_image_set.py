import os
from pathlib import Path

from config_parameters import *


class RestrictedFilePathCreatorTF:
    """
        Handles the case where we want to train on a subset of data.
    """

    def __init__(self, ratio, files_path, data_path):
        self.ratio = ratio
        self.files_path = files_path
        self.data_path = data_path

        self.new_file_names_path = None
        self.n_samples = 0

        if ratio == 0.0:
            self.__create_file_names_from_existing_data()
        elif 0.0 < self.ratio < 1.0:
            self.__create_split_from_file_names()
        else:
            raise ValueError("Class should only be called when 0.0 <= ratio < 1.0")

    def __create_file_names_from_existing_data(self):
        with open(self.files_path, 'r') as f:
            all_file_names = f.read().splitlines()

        all_file_names_with_correct_extension = self.__apply_correct_extension_to_set(all_file_names)

        to_remove = []
        for pos, line in enumerate(all_file_names_with_correct_extension):
            single_left_path = os.path.join(self.data_path, line.split(' ')[0])
            if not os.path.exists(single_left_path):
                to_remove.append(pos)

        for pos in to_remove[::-1]:
            all_file_names.pop(pos)

        all_file_names = '\n'.join(all_file_names)

        new_file_path = '/'.join(self.files_path.split('/')[:-1]) + TEMP_FILENAMES_PATH
        with open(new_file_path, 'w') as f:
            f.write(all_file_names)

        self.new_file_names_path = new_file_path
        self.n_samples = len(all_file_names.splitlines())

    def __create_split_from_file_names(self):
        with open(self.files_path, 'r') as f:
            all_file_names = f.read().splitlines()

        split_at = int(self.ratio * len(all_file_names))
        split_file_names = all_file_names[:split_at]
        file_name_string = '\n'.join(split_file_names)

        new_file_path = '/'.join(self.files_path.split('/')[:-1]) + TEMP_FILENAMES_PATH
        with open(new_file_path, 'w') as f:
            f.write(file_name_string)

        self.new_file_names_path = new_file_path
        self.n_samples = len(file_name_string.splitlines())

    def __apply_correct_extension_to_set(self, file_names):
        all_files = list(Path(self.data_path).rglob("*.[pj][np][g]"))
        exts_set = {(str(ele).split('/')[-1]).split('.')[-1] for ele in all_files}
        if 'png' in exts_set:
            new_file_names = [[os.path.splitext(ele)[0] + '.png' for ele in line.split(' ')] for line in file_names]
            return [' '.join(line) for line in new_file_names]
        return file_names


class RestrictedFilePathCreatorTorch:
    """
        Handles the case where we want to train on a subset of data.
    """

    def __init__(self, ratio, all_file_paths, data_dir_path):
        self.ratio = ratio
        self.all_files_paths = all_file_paths
        self.data_dir_path = data_dir_path

        if ratio == 0.0:
            self.__create_file_names_from_existing_data()
        elif 0.0 < self.ratio < 1.0:
            self.__create_split_from_file_names()
        else:
            raise ValueError("Class should only be called when 0.0 <= ratio < 1.0")

    def __create_file_names_from_existing_data(self):
        all_file_names_with_correct_extension = self.__apply_correct_extension_to_set(self.all_files_paths)

        to_remove = []
        for pos, line in enumerate(all_file_names_with_correct_extension):
            single_left_path = os.path.join(self.data_dir_path, line[0])

            if single_left_path == '/home/rick/data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data/0000000090.jpg':
                print('hey')

            if not os.path.exists(single_left_path):
                to_remove.append(pos)

        for pos in to_remove[::-1]:
            self.all_files_paths.pop(pos)

    def __create_split_from_file_names(self):
        split_at = int(self.ratio * len(self.all_files_paths))
        self.all_files_paths = self.all_files_paths[:split_at]

    def __apply_correct_extension_to_set(self, file_names):
        all_files = list(Path(self.data_dir_path).rglob("*.[pj][np][g]"))
        exts_set = {os.path.splitext(str(name))[1] for name in all_files
                    if not any(substring in str(name) for substring in ['training', 'testing', 'cs_disparity'])}
        if 'png' in exts_set:
            new_file_names = [[os.path.splitext(ele)[0] + '.png' for ele in line] for line in file_names]
            return new_file_names
        return file_names


def check_if_all_images_are_present(data_set, data_dir_path):
    set_names = ['train', 'val', 'test']

    if data_set == 'kitti':
        data_set_path = KITTI_PATH
    elif data_set == 'eigen':
        data_set_path = EIGEN_PATH
    else:
        data_set_path = CITYSCAPES_PATH

    all_file_names_paths = []
    for set_name in set_names:
        full_path_to_set = data_set_path.format(set_name)
        with open(full_path_to_set, 'r') as f:
            data_set_names = f.read()
            data_set_names = data_set_names.splitlines()
        for names in data_set_names:
            names = names.split(' ')
            all_file_names_paths.extend(names)

    total_length = len(all_file_names_paths)
    number_not_present = 0
    for file_path in all_file_names_paths:
        full_file_path = os.path.join(data_dir_path, file_path)
        if os.path.exists(full_file_path):
            number_not_present += 1

    print("-- Dir check for {}: --".format(data_set))
    print("{} out of {} images present".format(number_not_present, total_length))
    return


def get_present_images_from_list(lst_of_paths, data_dir):
    to_remove = []
    for pos, line in enumerate(lst_of_paths):
        line = line.split()
        single_left_path = os.path.join(data_dir, line[0])
        if not os.path.exists(single_left_path):
            to_remove.append(pos)

    for pos in to_remove[::-1]:
        lst_of_paths.pop(pos)
    return lst_of_paths
