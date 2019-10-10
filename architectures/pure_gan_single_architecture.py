import torch
import torch.optim as optim

from architectures import BaseArchitecture
from networks import define_D, define_generator_loss, define_discriminator_loss, to_device
from utils.image_pool import ImagePool


class ZeroLoss:
    @staticmethod
    def item():
        return 0.0


class PureGanSingleArchitecture(BaseArchitecture):

    def __init__(self, args):
        super().__init__(args)

        if args.mode == 'train':
            self.D = define_D(args)
            self.D = self.D.to(self.device)

            self.fake_right_pool = ImagePool(50)

            self.criterionMonoDepth = define_generator_loss(args)
            self.criterionMonoDepth = self.criterionMonoDepth.to(self.device)

            self.criterionGAN = define_discriminator_loss(args)
            self.criterionGAN = self.criterionGAN.to(self.device)

        # Load the correct networks, depending on which mode we are in.
        if args.mode == 'train':
            self.model_names = ['G', 'D']
            self.optimizer_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.loss_names = ['G', 'D']

        # We do Resume Training for this architecture.
        if args.resume == '':
            pass
        else:
            self.load_checkpoint(load_optim=False)

        if args.mode == 'train':
            # After resuming, set new optimizers.
            self.optimizer_G = optim.SGD(self.G.parameters(), lr=args.learning_rate)
            self.optimizer_D = optim.SGD(self.D.parameters(), lr=args.learning_rate)

            # Reset epoch.
            self.start_epoch = 0

        self.trainG = True
        self.count_trained_G = 0
        self.count_trained_D = 0
        self.regime = args.resume_regime

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data['left_image']
        self.right = self.data['right_image']

    def forward(self):
        self.disps = self.G(self.left)

        # Prepare disparities
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in self.disps]
        self.disp_right_est = disp_right_est[0]

        self.fake_right = self.criterionMonoDepth.generate_image_right(self.left, self.disp_right_est)

    def backward_D(self):
        # Fake
        fake_pool = self.fake_right_pool.query(self.fake_right)
        pred_fake = self.D(fake_pool.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.D(self.right)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # G should fake D
        pred_fake = self.D(self.fake_right)
        self.loss_G = self.criterionGAN(pred_fake, True) * self.args.discriminator_w
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Update D.
        if self.regime == [0, 0] or not self.trainG:
            self.set_requires_grad(self.D, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # Switch training, if regimes counts for D have been met.
        if self.regime != [0, 0] and not self.trainG:
            self.loss_G = ZeroLoss
            self.count_trained_D += 1
            if self.count_trained_D >= self.regime[1]:
                self.count_trained_D = 0
                self.trainG = True

        # Update G.
        if self.regime == [0, 0] or self.trainG:
            self.set_requires_grad(self.D, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        # Switch training, if regimes counts for D have been met.
        if self.regime != [0, 0] and self.trainG:
            self.loss_D = ZeroLoss
            self.count_trained_G += 1
            if self.count_trained_G >= self.regime[0]:
                self.count_trained_G = 0
                self.trainG = False

    def update_learning_rate(self, epoch, learning_rate):
        """ Sets the learning rate to the initial LR
            decayed by 2 every 10 epochs after 30 epochs.
        """
        if self.args.adjust_lr:
            if 30 <= epoch < 40:
                lr = learning_rate / 2
            elif epoch >= 40:
                lr = learning_rate / 4
            else:
                lr = learning_rate
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr

    def get_untrained_loss(self):
        # -- Generator
        fake_G_right = self.D(self.fake_right)
        loss_G = self.criterionGAN(fake_G_right, True) * self.args.discriminator_w

        # -- Discriminator
        loss_D_fake = self.criterionGAN(self.D(self.fake_right), False)
        loss_D_real = self.criterionGAN(self.D(self.right), True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return {'G': loss_G.item(), 'D': loss_D.item()}

    @property
    def architecture(self):
        return 'Pure Single GAN Architecture'
