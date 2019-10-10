import torch.nn as nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, num_out=64):
        super(SimpleDiscriminator, self).__init__()
        self.num_out = num_out

        main = nn.Sequential(
            nn.Conv2d(3, self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_out, 2 * self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.num_out, 4 * self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * self.num_out, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * self.num_out)
        output = self.linear(output)
        return output
