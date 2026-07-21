import torch
import torch.nn as nn
import torch.nn.init as init

class CONV3D_BLOCK(nn.Module):
    def __init__(self,in_channel,out_channel,normal=None,activation=None,ksize=3,stride=1,pad=1):
        super(CONV3D_BLOCK,self).__init__()
        layers = []
        conv = nn.Conv3d(in_channel,out_channel,kernel_size=ksize,stride=stride,padding=pad,padding_mode='reflect')
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
    def __init__(self,in_channel,out_channel,upmode='CT'):
        super(UPSAMPLE,self).__init__()
        model = []
        if upmode == 'CT':
            model += [nn.ConvTranspose3d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)]
        elif upmode == 'nearest':
            model += [nn.Upsample(scale_factor=2, mode='nearest')]
            model += [nn.Conv3d(in_channel,out_channel,kernel_size=3,stride=1,padding=1,padding_mode='reflect')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
'''  
class UPSAMPLE(nn.Module):
    def __init__(self,in_channel,upmode = 'CT'):
        super(UPSAMPLE,self).__init__()
        model = []
        if upmode == 'CT':
            model += [ nn.ConvTranspose3d(in_channel, in_channel,
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       nn.BatchNorm3d(in_channel),
                       nn.LeakyReLU(0.2, inplace=True) ]
        elif upmode == 'nearest':
            model += [nn.Upsample(scale_factor=2, mode='nearest')]
            model += [CONV3D_BLOCK(in_channel, in_channel, normal='BN', activation='LeakyReLU')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
'''

class CBAM(nn.Module):
    def __init__(self,channel,SK_size=3):
        super(CBAM,self).__init__()
        self.channel_att = CAM(channel)
        self.spatial_att = SAM(ksize=SK_size)

    def forward(self,x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CAM(nn.Module):
    def __init__(self,channel,reduction=16):
        super(CAM,self).__init__()
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(nn.Conv3d(channel,channel//reduction,kernel_size=1,bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv3d(channel//reduction,channel,kernel_size=1,bias=False),)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x

class SAM(nn.Module):
    def __init__(self,ksize=7):
        super(SAM,self).__init__()
        self.conv = nn.Conv3d(2,1,kernel_size=ksize,
                              padding=ksize//2,padding_mode='reflect',bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out,_ = torch.max(x, dim=1,keepdim=True)
        avg_out = torch.mean(x, dim=1,keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class SFAM(nn.Module):
    def __init__(self,ksize=7):
        super(SFAM,self).__init__()
        self.conv_k7 = nn.Conv3d(2,1,kernel_size=7,
                              padding=7//2,padding_mode='reflect',bias=False)
        self.conv_k5 = nn.Conv3d(2, 1, kernel_size=5,
                                 padding=5 // 2, padding_mode='reflect', bias=False)
        self.conv_k3 = nn.Conv3d(2, 1, kernel_size=3,
                                 padding=3 // 2, padding_mode='reflect', bias=False)
        self.conv_out = nn.Conv3d(3, 1, kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out,_ = torch.max(x, dim=1,keepdim=True)
        avg_out = torch.mean(x, dim=1,keepdim=True)

        spatial_out_k7 = self.conv_k7(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k5 = self.conv_k5(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k3 = self.conv_k3(torch.cat([max_out, avg_out],dim=1))
        spatial_out = self.sigmoid(self.conv_out(torch.cat([spatial_out_k7, spatial_out_k5,spatial_out_k3], dim=1)))
        x = spatial_out * x
        return x

class SFAM2(nn.Module):
    def __init__(self,ksize=7):
        super(SFAM2,self).__init__()
        self.conv_k7 = nn.Conv3d(2,1,kernel_size=7,
                              padding=7//2,padding_mode='reflect',bias=False)
        self.conv_k5 = nn.Conv3d(2, 1, kernel_size=5,
                                 padding=5 // 2, padding_mode='reflect', bias=False)
        self.conv_k3 = nn.Conv3d(2, 1, kernel_size=3,
                                 padding=3 // 2, padding_mode='reflect', bias=False)
        # self.conv_out = nn.Conv3d(3, 1, kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out,_ = torch.max(x, dim=1,keepdim=True)
        avg_out = torch.mean(x, dim=1,keepdim=True)

        spatial_out_k7 = self.conv_k7(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k5 = self.conv_k5(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k3 = self.conv_k3(torch.cat([max_out, avg_out],dim=1))
        spatial_out = self.sigmoid(spatial_out_k7+spatial_out_k5+spatial_out_k3)
        x = spatial_out * x
        return x

class SFAM3(nn.Module):
    def __init__(self,ksize=7):
        super(SFAM3,self).__init__()
        self.conv_k7 = nn.Conv3d(2,1,kernel_size=(5,7,7),
                              padding=(2,3,3),padding_mode='reflect',bias=False)
        self.conv_k5 = nn.Conv3d(2, 1, kernel_size=(3,5,5),
                                 padding=(1,2,2), padding_mode='reflect', bias=False)
        self.conv_k3 = nn.Conv3d(2, 1, kernel_size=(1,3,3),
                                 padding=(0,1,1), padding_mode='reflect', bias=False)
        # self.conv_out = nn.Conv3d(3, 1, kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out,_ = torch.max(x, dim=1,keepdim=True)
        avg_out = torch.mean(x, dim=1,keepdim=True)

        spatial_out_k7 = self.conv_k7(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k5 = self.conv_k5(torch.cat([max_out, avg_out], dim=1))
        spatial_out_k3 = self.conv_k3(torch.cat([max_out, avg_out],dim=1))
        spatial_out = self.sigmoid(spatial_out_k7+spatial_out_k5+spatial_out_k3)
        x = spatial_out * x
        return x

class CSFM(nn.Module):
    def __init__(self,channel):
        super(CSFM,self).__init__()
        self.channel_att = CAM(channel)
        # self.spatial_att = SFAM() #SFAM1
        # self.spatial_att = SFAM2()  #SFAM2
        self.spatial_att = SFAM3()  # SFAM3

    def forward(self,x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x




