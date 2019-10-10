import torch
import torch.optim as optim

from architectures import BaseArchitecture
from networks import define_D, define_generator_loss, define_discriminator_loss, to_device
from utils.image_pool import ImagePool


class VanillaGanSingleArchitecture(BaseArchitecture):

    def __init__(self, args):
        super().__init__(args)

        if args.mode == 'train':
            self.D = define_D(args)
            self.D = self.D.to(self.device)

            self.fake_right_pool = ImagePool(50)

            self.criterion = define_generator_loss(args)
            self.criterion = self.criterion.to(self.device)
            self.criterionGAN = define_discriminator_loss(args)
            self.criterionGAN = self.criterionGAN.to(self.device)

            self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.learning_rate)
            self.optimizer_D = optim.SGD(self.D.parameters(), lr=args.learning_rate)

        # Load the correct networks, depending on which mode we are in.
        if args.mode == 'train':
            self.model_names = ['G', 'D']
            self.optimizer_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.loss_names = ['G', 'G_MonoDepth', 'G_GAN', 'D']
        self.losses = {}

        if self.args.resume:
            self.load_checkpoint()

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

        self.fake_right = self.criterion.generate_image_right(self.left, self.disp_right_est)

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
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_MonoDepth = self.criterion(self.disps, [self.left, self.right])

        self.loss_G = self.loss_G_GAN * self.args.discriminator_w + self.loss_G_MonoDepth
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Update D.
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Update G.
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

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
        loss_G_MonoDepth = self.criterion(self.disps, [self.left, self.right])
        fake_G_right = self.D(self.fake_right)
        loss_G_GAN = self.criterionGAN(fake_G_right, True)
        loss_G = loss_G_GAN * self.args.discriminator_w + loss_G_MonoDepth

        # -- Discriminator
        loss_D_fake = self.criterionGAN(self.D(self.fake_right), False)
        loss_D_real = self.criterionGAN(self.D(self.right), True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return {'G': loss_G.item(), 'G_MonoDepth': loss_G_MonoDepth.item(),
                'G_GAN': loss_G_GAN.item(), 'D': loss_D.item()}

    @property
    def architecture(self):
        return 'Single GAN Architecture'
