# Import project files
from options import EvaluateOptions
from utils.evaluation_utils import *
from config_parameters import *


def evaluate_eigen(args, verbose=True):
    pred_disparities = np.load(args.predicted_disp_path)

    test_files = sorted(read_text_lines(EIGEN_PATH.format('test')))
    num_samples = len(test_files)

    assert_str = "Only {} disparities recovered out of required {}".format(len(pred_disparities), num_samples)
    assert len(pred_disparities) == num_samples, assert_str

    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.data_dir)

    gt_depths = []
    pred_depths = []
    for t_id in range(num_samples):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
        gt_depths.append(depth.astype(np.float32))

        disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]),
                               interpolation=cv2.INTER_LINEAR)
        disp_pred = disp_pred * disp_pred.shape[1]

        # need to convert from disparity to depth
        focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
        depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape

            # crop used by Garg ECCV16
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            if args.garg_crop:
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            # crop we found by trial and error to reproduce Eigen NIPS14 results
            elif args.eigen_crop:
                crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                 0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                        pred_depth[mask])

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms',
                                                                              'log_rms', 'a1', 'a2', 'a3'))
        print("{:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}".format(abs_rel.mean(), sq_rel.mean(),
                                                                                            rms.mean(), log_rms.mean(),
                                                                                            a1.mean(), a2.mean(), a3.mean()))

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()


def evaluate_kitti(args, verbose=True):
    pred_disparities = np.load(args.predicted_disp_path)
    num_samples = 200

    assert_str = "Only {} disparities recovered out of required {}".format(len(pred_disparities), num_samples)
    assert len(pred_disparities) == num_samples, assert_str

    gt_disparities = load_gt_disp_kitti(args.data_dir)
    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                        pred_depth[mask])

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                                                      'd1_all', 'a1', 'a2', 'a3'))
        print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                                      sq_rel.mean(),
                                                                                                      rms.mean(),
                                                                                                      log_rms.mean(),
                                                                                                      d1_all.mean(),
                                                                                                      a1.mean(), a2.mean(),
                                                                                                      a3.mean()))
    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()


def evaluate_cityscapes(args):
    """ Please note that allocating [1525 x 1024 x 2048] twice takes up 20+ GB of RAM.
    """
    # Set args.min_depth to 3, because there are no lower depth values in gt_depth
    # maps of CityScapes, it seems.
    args.min_depth = 3.0

    pred_disparities = np.load(args.predicted_disp_path)
    num_samples = 1525

    assert_str = "Only {} disparities recovered out of required {}".format(len(pred_disparities), num_samples)
    assert len(pred_disparities) == num_samples, assert_str

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    if args.save_best_worst:
        # Make bins for depth values. GT depths are between 1 and 80.
        lin_bins = np.linspace(0.0, 80.0, 81)
        bin_counts_gt = np.zeros(shape=(num_samples, 80))
        bin_counts_pred = np.zeros(shape=(num_samples, 80))

    for i in range(num_samples):
        gt_depth, mask = load_gt_depth_cityscapes(args.data_dir, i)
        height, width = gt_depth.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
        pred_depth = (0.209313 * 2262.52) / pred_disp

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        # Also bound depth maps to a max of 80, because the maxes lie around 470. Mention this carefully in
        # thesis. Metrics give very other results than Eigen split. Depth values should be proportional.
        gt_depth[gt_depth < args.min_depth] = args.min_depth
        gt_depth[gt_depth > args.max_depth] = args.max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                        pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms',
                                                                          'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(),
                                                                                        rms.mean(), log_rms.mean(),
                                                                                        a1.mean(), a2.mean(), a3.mean()))
    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()


if __name__ == '__main__':
    parser = EvaluateOptions()
    args = parser.parse()

    if args.split == 'eigen':
        evaluate_eigen(args)
    elif args.split == 'kitti':
        evaluate_kitti(args)
    elif args.split == 'cityscapes':
        evaluate_cityscapes(args)
    else:
        print("Split not recognised.")
