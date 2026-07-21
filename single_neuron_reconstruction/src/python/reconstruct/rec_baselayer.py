import torch.nn as nn

class conv_block_2D(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm2d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



