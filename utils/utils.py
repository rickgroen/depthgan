import numpy as np


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)

    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def print_epoch_update(epoch, time, losses):
    # Print update for this epoch.
    train_losses = losses[epoch]['train']
    val_losses = losses[epoch]['val']

    train_loss_string = 'train: \t['
    val_loss_string = 'val:   \t['
    for key in train_losses.keys():
        train_loss_string += "  {}: {:.4f}  ".format(key, train_losses[key])
        val_loss_string += "  {}: {:.4f}  ".format(key, val_losses[key])

    update_print_statement = 'Epoch: {}\t | train: {:.2f}\t | val: {:.2f}\t | time: {:.2f}\n  {}]\n  {}]'
    print(update_print_statement.format(epoch, losses[epoch]['train']['G'], losses[epoch]['val']['G'],
                                        time, train_loss_string, val_loss_string))
    return


def pre_validation_update(val_losses):
    val_loss_string = 'Pre-training val losses:\t['
    for key in val_losses.keys():
        val_loss_string += "  {}: {:.4f}  ".format(key, val_losses[key])
    val_loss_string += ']'
    print(val_loss_string)
    return
