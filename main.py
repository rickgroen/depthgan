import time
import torch
import os

# Project imports
from utils import *
from options import MainOptions
from data_loader import prepare_dataloader
from architectures import create_architecture


def train(args):
    """ Function used for training any of the architectures, given an input parse.
    """

    def validate(epoch):
        model.to_test()

        disparities = np.zeros((val_n_img, 256, 512), dtype=np.float32)
        model.set_new_loss_item(epoch, train=False)

        # For a WGAN architecture we need to access gradients.
        if 'wgan' not in args.architecture:
            torch.set_grad_enabled(False)

        for i, data in enumerate(val_loader):
            # Get the losses for the model for this epoch.
            model.set_input(data)
            model.forward()
            model.add_running_loss_val(epoch)

        if 'wgan' not in args.architecture:
            torch.set_grad_enabled(True)

        # Store the running loss for the validation images.
        model.make_running_loss(epoch, val_n_img, train=False)
        return

    n_img, loader = prepare_dataloader(args, 'train')
    val_n_img, val_loader = prepare_dataloader(args, 'val')

    model = create_architecture(args)
    model.set_data_loader(loader)

    if not args.resume:
        # We keep track of the aggregated losses per epoch in a dict. For
        # now the pre-training train loss is set to zero. The pre-training
        # validation loss will be computed.
        best_val_loss = float('Inf')

        # Compute loss per image (computation keeps number of images over
        # batch size in mind, to compensate for partial batches being forwarded.
        validate(-1)
        pre_validation_update(model.losses[-1]['val'])
    else:
        best_val_loss = min([model.losses[epoch]['val']['G'] for epoch in model.losses.keys()])

    running_val_loss = 0.0

    for epoch in range(model.start_epoch, args.epochs):
        model.update_learning_rate(epoch, args.learning_rate)

        c_time = time.time()
        model.to_train()
        model.set_new_loss_item(epoch)

        # Run a single training epoch. Generalizes to WGAN variants as well.
        model.run_epoch(epoch, n_img)

        # The validate can return either a dictionary with metrics or None.
        validate(epoch)

        # Print an update of training, val losses. Possibly also do full evaluation of depth maps.
        print_epoch_update(epoch, time.time() - c_time, model.losses)

        # Make a checkpoint, so training can be resumed.
        running_val_loss = model.losses[epoch]['val']['G']
        is_best = running_val_loss < best_val_loss
        if is_best:
            best_val_loss = running_val_loss
        model.save_checkpoint(epoch, is_best, best_val_loss)

    print('Finished Training. Best validation loss:\t{:.3f}'.format(best_val_loss))

    # Save the model of the final epoch. If another model was better, also save it separately as best.
    model.save_networks('final')
    if running_val_loss != best_val_loss:
        model.save_best_networks()

    model.save_losses()


def test(args):
    """ Function to test the architecture by saving disparities to the output directory
    """
    # Since it is clear post-processing is better in all runs I have done, I will only
    # save post-processed results. Unless explicitly stated otherwise.
    # Also for Pilzer, the disparities are already post-processed by their own FuseNet.
    do_post_processing = args.postprocessing and 'pilzer' not in args.architecture

    input_height = args.input_height
    input_width = args.input_width

    output_directory = args.output_dir
    n_img, test_loader = prepare_dataloader(args, 'test')

    model = create_architecture(args)
    which_model = 'final' if args.load_final else 'best'
    model.load_networks(which_model)
    model.to_test()

    disparities = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    inference_time = 0.0

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 100 == 0 and i != 0:
                print('Testing... Now at image: {}'.format(i))

            t_start = time.time()
            # Do a forward pass
            disps = model.fit(data)
            # Some architectures output a single disparity, not a tuple of 4 disparities.
            disps = disps[0][:, 0, :, :] if isinstance(disps, tuple) else disps.squeeze()

            if do_post_processing:
                disparities[i] = post_process_disparity(disps.cpu().numpy())
            else:
                disp = disps.unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
            t_end = time.time()
            inference_time += (t_end - t_start)

    if args.test_time:
        test_time_message = 'Inference took {:.4f} seconds. That is {:.2f} imgs/s or {:.6f} s/img.'
        print(test_time_message.format(inference_time, (n_img / inference_time), 1.0 / (n_img / inference_time)))

    disp_file_name = 'disparities_{}_{}.npy'.format(args.dataset, model.name)
    full_disp_path = os.path.join(output_directory, disp_file_name)

    if os.path.exists(full_disp_path):
        print('Overwriting disparities at {}...'.format(full_disp_path))
    np.save(full_disp_path, disparities)
    print('Finished Testing')


def main():
    parser = MainOptions()
    args = parser.parse()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'verify-data':
        from utils.reduce_image_set import check_if_all_images_are_present
        check_if_all_images_are_present('kitti', args.data_dir)
        check_if_all_images_are_present('eigen', args.data_dir)
        check_if_all_images_are_present('cityscapes', args.data_dir)


if __name__ == '__main__':
    main()

    # Do an Arnold.
    print("YOU ARE TERMINATED!")
