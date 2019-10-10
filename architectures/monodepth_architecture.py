import torch
import torch.optim as optim

from architectures import BaseArchitecture
from networks import to_device, define_generator_loss


class MonoDepthArchitecture(BaseArchitecture):

    def __init__(self, args):
        super().__init__(args)

        if args.mode == 'train':
            self.criterion = define_generator_loss(args)
            self.criterion = self.criterion.to(self.device)
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.learning_rate)

        # Load the correct networks, depending on which mode we are in.
        if args.mode == 'train':
            self.model_names = ['G']
            self.optimizer_names = ['G']
        else:
            self.model_names = ['G']

        self.loss_names = ['G']
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

    def backward(self):
        self.loss_G = self.criterion(self.disps, [self.left, self.right])
        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward()
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

    def get_untrained_loss(self):
        loss_G = self.criterion(self.disps, [self.left, self.right])
        return {'G': loss_G.item()}

    @property
    def architecture(self):
        return 'Mono Depth'
