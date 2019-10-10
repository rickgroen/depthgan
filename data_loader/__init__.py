import importlib

from torch.utils.data import DataLoader, ConcatDataset
from config_parameters import KITTI_PATH, EIGEN_PATH, CITYSCAPES_PATH
from data_loader.data_loader import StreetMonoLoader as StreetLoader


def import_transforms():
    transforms_filename = 'data_loader.transforms'
    transforms_lib = importlib.import_module(transforms_filename)

    method_name_kitti = 'image_transforms_kitti'
    transforms_kitti = getattr(transforms_lib, method_name_kitti)
    method_name_cityscapes = 'image_transforms_cityscapes'
    transforms_cityscapes = getattr(transforms_lib, method_name_cityscapes)

    if transforms_kitti is None or transforms_cityscapes is None:
        raise ValueError('No transforms method was found in {}.py,'.format(transforms_filename))
    return transforms_kitti, transforms_cityscapes


def obtain_correct_path_to_files(args, data_mode):
    """ Set correct dataset
    """
    if args.dataset == 'kitti' and args.split == 'kitti':
        path_to_dataset_file_names = KITTI_PATH
        return path_to_dataset_file_names.format(data_mode)
    elif args.dataset == 'kitti' and args.split == 'eigen':
        path_to_dataset_file_names = EIGEN_PATH
        return path_to_dataset_file_names.format(data_mode)
    elif args.dataset == 'cityscapes':
        path_to_dataset_file_names = CITYSCAPES_PATH
        return path_to_dataset_file_names.format(data_mode)
    elif args.dataset == 'both' and args.split == 'kitti':
        path_to_dataset_file_names_kitti = KITTI_PATH.format(data_mode)
        path_to_dataset_file_names_cityscapes = CITYSCAPES_PATH.format(data_mode)
        return path_to_dataset_file_names_kitti, path_to_dataset_file_names_cityscapes
    elif args.dataset == 'both' and args.split == 'eigen':
        path_to_dataset_file_names_kitti = EIGEN_PATH.format(data_mode)
        path_to_dataset_file_names_cityscapes = CITYSCAPES_PATH.format(data_mode)
        return path_to_dataset_file_names_kitti, path_to_dataset_file_names_cityscapes
    else:
        raise ValueError("There is no code to run CityScape data yet.")


def prepare_dataloader(args, data_mode, verbose=True):
    # According to task, import right transforms.
    image_transforms_kitti, image_transforms_cityscapes = import_transforms()

    if args.dataset == 'kitti':
        data_paths_file = obtain_correct_path_to_files(args, data_mode)
        data_transform = image_transforms_kitti(
            mode=data_mode,
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            size=(args.input_height, args.input_width))

        dataset = StreetLoader(args.data_dir, data_paths_file, data_mode,
                               args.train_ratio, transform=data_transform)
    elif args.dataset == 'cityscapes':
        data_paths_file = obtain_correct_path_to_files(args, data_mode)
        data_transform = image_transforms_cityscapes(
            mode=data_mode,
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            size=(args.input_height, args.input_width))

        dataset = StreetLoader(args.data_dir, data_paths_file, data_mode,
                               args.train_ratio, transform=data_transform)
    else:
        data_paths_file = obtain_correct_path_to_files(args, data_mode)
        data_paths_file_kitti = data_paths_file[0]
        data_paths_file_cityscapes = data_paths_file[1]

        data_transform_kitti = image_transforms_kitti(
            mode=data_mode,
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            size=(args.input_height, args.input_width))
        data_transform_cityscapes = image_transforms_cityscapes(
            mode=data_mode,
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            size=(args.input_height, args.input_width))

        dataset_kitti = StreetLoader(args.data_dir, data_paths_file_kitti, data_mode,
                                     args.train_ratio, transform=data_transform_kitti)
        dataset_cityscapes = StreetLoader(args.data_dir, data_paths_file_cityscapes, data_mode,
                                          args.train_ratio, transform=data_transform_cityscapes)
        dataset = ConcatDataset([dataset_kitti, dataset_cityscapes])

    n_img = len(dataset)
    if verbose:
        print('Loaded a dataset with {} images'.format(n_img))

    if data_mode == 'train':
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_threads,
                            pin_memory=True)
    # If val or test, then feed unshuffled raw data.
    else:
        loader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1,
                            pin_memory=True)
    return n_img, loader
