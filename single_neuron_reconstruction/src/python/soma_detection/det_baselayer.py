import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CONV2D_BLOCK(nn.Module):
    def __init__(self,in_channel,out_channel,normal=None,activation=None,stride=1,k_size=3):
        super(CONV2D_BLOCK,self).__init__()
        layers = []
        conv = nn.Conv2d(in_channel,out_channel,kernel_size=k_size,stride=stride,padding=k_size//2,padding_mode='reflect')
        layers.append(conv)
        #normal_layer
        if normal == 'BN':
            layers.append(nn.BatchNorm2d(out_channel))
        elif normal == 'IN':
            layers.append(nn.InstanceNorm2d(out_channel))
        #activation
        if activation == 'ReLU':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'Tanh':
            layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)

class CONV3D_BLOCK(nn.Module):
    def __init__(self,in_channel,out_channel,normal=None,activation=None,stride=1,k_size=3):
        super(CONV3D_BLOCK,self).__init__()
        layers = []
        conv = nn.Conv3d(in_channel,out_channel,kernel_size=k_size,stride=stride,padding=k_size//2,padding_mode='reflect')
        layers.append(conv)
        #normal_layer
        if normal == 'BN':
            layers.append(nn.BatchNorm3d(out_channel))
        elif normal == 'IN':
            layers.append(nn.InstanceNorm3d(out_channel))
        #activation
        if activation == 'ReLU':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'Tanh':
            layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)

class UPSAMPLE(nn.Module):
    def __init__(self,in_channel,mode = 'CT'):
        super(UPSAMPLE,self).__init__()
        model = []
        if mode == 'CT':
            model += [ nn.ConvTranspose3d(in_channel, in_channel,
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       nn.BatchNorm3d(in_channel),
                       nn.LeakyReLU(0.2, inplace=True) ]
        elif mode == 'nearest':
            model += [nn.Upsample(scale_factor=2, mode='nearest')]
            model += [CONV3D_BLOCK(in_channel, in_channel, normal='BN', activation='LeakyReLU')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class SAPF(nn.Module):
    def __init__(self,channels=512):
        super(SAPF, self).__init__()
        self.conv3x3 = nn.Conv3d(in_channels=channels, out_channels=channels,kernel_size=3, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm3d(channels),
                                 nn.BatchNorm3d(channels),
                                 nn.BatchNorm3d(channels)])

        self.layer1 = nn.Sequential(
            CONV3D_BLOCK(channels*2,channels//2,normal='BN', activation='LeakyReLU'),
            nn.Conv3d(in_channels=channels//2, out_channels=2, kernel_size=3,padding=1,padding_mode='reflect',bias=False),
        )

        self.layer2 = nn.Sequential(
            CONV3D_BLOCK(channels*2, channels // 2, normal='BN', activation='LeakyReLU'),
            nn.Conv3d(in_channels=channels // 2, out_channels=2, kernel_size=3,padding=1,padding_mode='reflect',bias=False),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.leakrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self, x):
        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv3d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv3d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.bn[2](branches_3)

        feat = torch.cat([branches_1, branches_2], dim=1)
        # feat=feat_cat.detach()
        att = self.layer1(feat)
        att = F.softmax(att, dim=1) #[batch,2,d,h,w]

        att_1 = att[:, 0, :, :,:].unsqueeze(1)
        att_2 = att[:, 1, :, :,:].unsqueeze(1)
        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)
        # feat=feat_cat.detach()
        att1 = self.layer2(feat1)
        att1 = F.softmax(att1, dim=1) #[batch,2,d,h,w]

        att_1_2 = att1[:, 0, :, :,:].unsqueeze(1)
        att_3 = att1[:, 1, :, :,:].unsqueeze(1)
        fusion_1_2_3 = att_1_2 * fusion_1_2 + att_3 * branches_3
        # print(fusion_1_2_3.shape,x.shape)
        ax = self.leakrelu(self.gamma * fusion_1_2_3 + (1 - self.gamma) * x)
        return ax