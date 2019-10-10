import os
from PIL import Image

from torch.utils.data import Dataset
from utils.reduce_image_set import RestrictedFilePathCreatorTorch


class StreetMonoLoader(Dataset):
    def __init__(self, data_dir, path_to_file_paths, mode, train_ratio, transform=None):
        with open(path_to_file_paths, 'r') as f:
            all_paths = f.read().splitlines()
            all_split_paths = [pair.split(' ') for pair in all_paths]

        # If we want a subset of data, now is the time to select it.
        if train_ratio != 1.0:
            restricted_creator = RestrictedFilePathCreatorTorch(train_ratio, all_split_paths, data_dir)
            all_split_paths = restricted_creator.all_files_paths

        left_right_path_lists = [lst for lst in zip(*all_split_paths)]

        left_fnames = list(left_right_path_lists[0])
        self.left_paths = sorted([os.path.join(data_dir, fname) for fname in left_fnames])

        if mode == 'train' or mode == 'val':
            right_fnames = list(left_right_path_lists[1])
            self.right_paths = sorted([os.path.join(data_dir, fname) for fname in right_fnames])

            assert len(self.right_paths) == len(self.left_paths), "Paths file might be corrupted."

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
        # If we have a validation set, we need both the left, right images, but also the test-like
        # left flipped image.
        elif self.mode == 'val':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}
            sample = self.transform(sample)
        # At test time we only use the left image, and its flipped counterpart.
        elif self.mode == 'test':
            if self.transform:
                left_image = self.transform(left_image)
            sample = {'left_image': left_image}
        else:
            raise ValueError('Mode {} not found in DataLoader'.format(self.mode))
        return sample
