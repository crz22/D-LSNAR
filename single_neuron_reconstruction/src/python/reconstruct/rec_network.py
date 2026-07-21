from reconstruct.rec_baselayer import *

class SPE_DNR(nn.Module):
    def __init__(self, NUM_ACTIONS, n):
        super(SPE_DNR, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n - 1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)

        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
        )
        self.dis_out = nn.Conv2d(64, 2 + 1, kernel_size=1, stride=1, padding=0)

        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # discriminator and radii
        out_dis = self.discriminator(out)
        out_dis = self.dis_out(out_dis)
        # direction
        out_dir = self.tracker(out)
        return out_dir, out_dis