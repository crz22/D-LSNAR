from segment.seg_baselayer import *

class UNet3d_CBAM1(nn.Module):
    def __init__(self,normal = 'BN',SK_size = 3):
        super(UNet3d_CBAM1, self).__init__()
        # [1,32,32,32]
        self.down_layer1 = nn.Sequential(CONV3D_BLOCK(1,32,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(32,64,normal,activation='LeakyReLU'),
                                   CBAM(64,SK_size))
        self.pool1 = nn.MaxPool3d(2)
        # [64,16,16,16]
        self.down_layer2 = nn.Sequential(CONV3D_BLOCK(64,64,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(64,128,normal,activation='LeakyReLU'),
                                   CBAM(128,SK_size))
        self.pool2 = nn.MaxPool3d(2)
        # [128,8,8,8]
        self.down_layer3 = nn.Sequential(CONV3D_BLOCK(128,128,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(128,256,normal,activation='LeakyReLU'),
                                   CBAM(256,SK_size))
        self.pool3 = nn.MaxPool3d(2)
        # [256,4,4,4]
        self.down_layer4 = nn.Sequential(CONV3D_BLOCK(256,256,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(256,512,normal,activation='LeakyReLU'),
                                   CBAM(512,SK_size))
        # [512,4,4,4]
        self.up1 = UPSAMPLE(512,512,upmode='CT')
        self.up_layer1 = nn.Sequential(CONV3D_BLOCK(512+256,256,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(256,256,normal,activation='LeakyReLU'))
        # [256,8,8,8]
        self.up2 = UPSAMPLE(256,256,upmode='CT')
        self.up_layer2 = nn.Sequential(CONV3D_BLOCK(256+128,128,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(128,128,normal,activation='LeakyReLU'))
        #[128,16,16,16]
        self.up3 = UPSAMPLE(128,128,upmode='CT')
        self.up_layer3 = nn.Sequential(CONV3D_BLOCK(128+64,64,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(64,64,normal,activation='LeakyReLU'))
        # [64,32,32,32]
        self.outlayer = CONV3D_BLOCK(64,2)

    def forward(self, x):
        y1 = self.down_layer1(x)
        d1 = self.pool1(y1)
        y2 = self.down_layer2(d1)
        d2 = self.pool2(y2)
        y3 = self.down_layer3(d2)
        d3 = self.pool3(y3)
        y4 = self.down_layer4(d3)

        u1 = self.up1(y4)
        z1 = self.up_layer1(torch.cat([u1,y3],1))
        u2 = self.up2(z1)
        z2 = self.up_layer2(torch.cat([u2,y2],1))
        u3 = self.up3(z2)
        z3 = self.up_layer3(torch.cat([u3,y1],1))

        out = self.outlayer(z3)
        return out

class MSFE(nn.Module):
    def __init__(self,normal = 'BN',K_SIZE=7):
        super(MSFE, self).__init__()
        # [1,32,32,32]
        self.down_layer1 = nn.Sequential(CONV3D_BLOCK(1,32,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(32,64,normal,activation='LeakyReLU'),
                                   CBAM(64,K_SIZE))
        self.pool1 = nn.MaxPool3d(2)
        # [64,16,16,16]
        self.down_layer2 = nn.Sequential(CONV3D_BLOCK(64,64,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(64,128,normal,activation='LeakyReLU'),
                                   CBAM(128,K_SIZE))
        self.pool2 = nn.MaxPool3d(2)
        # [128,8,8,8]
        self.down_layer3 = nn.Sequential(CONV3D_BLOCK(128,128,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(128,256,normal,activation='LeakyReLU'),
                                   CBAM(256,K_SIZE))
        self.pool3 = nn.MaxPool3d(2)
        # [256,4,4,4]
        self.down_layer4 = nn.Sequential(CONV3D_BLOCK(256,256,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(256,512,normal,activation='LeakyReLU'),
                                   CBAM(512,K_SIZE))

    def forward(self, x):
        y1 = self.down_layer1(x)
        d1 = self.pool1(y1)
        y2 = self.down_layer2(d1)
        d2 = self.pool2(y2)
        y3 = self.down_layer3(d2)
        d3 = self.pool3(y3)
        y4 = self.down_layer4(d3)

        return [y1,y2,y3,y4]

class PASD(nn.Module):
    def __init__(self,normal = 'BN'):
        super(PASD, self).__init__()
        # [512,4,4,4]
        self.up1 = UPSAMPLE(512, 512, upmode='CT')
        self.up_layer1 = nn.Sequential(CONV3D_BLOCK(512 + 256, 256, normal, activation='LeakyReLU'),
                                       CONV3D_BLOCK(256, 256, normal, activation='LeakyReLU'))
        # [256,8,8,8]
        self.up2 = UPSAMPLE(256, 256, upmode='CT')
        self.up_layer2 = nn.Sequential(CONV3D_BLOCK(256 + 128, 128, normal, activation='LeakyReLU'),
                                       CONV3D_BLOCK(128, 128, normal, activation='LeakyReLU'))
        # [128,16,16,16]
        self.up3 = UPSAMPLE(128, 128, upmode='CT')
        self.up_layer3 = nn.Sequential(CONV3D_BLOCK(128 + 64, 64, normal, activation='LeakyReLU'),
                                       CONV3D_BLOCK(64, 64, normal, activation='LeakyReLU'))
        # [64,32,32,32]
        self.outlayer = CONV3D_BLOCK(64, 2)
    def forward(self, mid_feature):
        y1, y2, y3, y4 = mid_feature

        u1 = self.up1(y4)
        z1 = self.up_layer1(torch.cat([u1, y3], 1))
        u2 = self.up2(z1)
        z2 = self.up_layer2(torch.cat([u2, y2], 1))
        u3 = self.up3(z2)
        z3 = self.up_layer3(torch.cat([u3, y1], 1))

        out = self.outlayer(z3)
        return out

class EFC(nn.Module):
    def __init__(self,normal = 'BN'):
        super(EFC, self).__init__()
        #[512,4,4,4]
        self.mlp_layer = nn.Sequential(CONV3D_BLOCK(512,256,normal,'LeakyReLU',ksize=1,pad=0),
                                       CONV3D_BLOCK(256,64,normal,'LeakyReLU',ksize=1,pad=0),)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Conv3d(128,2,1))

    def forward(self, x):
        # print(x.max(),x.min(),x.mean())
        x = self.mlp_layer(x)
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        out = self.classifier(torch.cat([x_max, x_avg], 1))
        return out.flatten(start_dim=1).view(-1,2)

class DTANET(nn.Module):
    def __init__(self,normal = 'BN',efc_norm = None,K_SZIE=3):
        super(DTANET, self).__init__()
        if efc_norm is None:
            efc_norm = normal
        self.msfe = MSFE(normal,K_SZIE)
        self.efc = EFC(efc_norm)
        self.PASD0 = PASD(normal)
        self.PASD1 = PASD(normal)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        mid_feature = self.msfe(x)
        class_out = self.efc(mid_feature[-1]) #[batch,2]
        class_out = self.softmax(class_out)
        class_out = torch.argmax(class_out, dim=1).view(-1,1)

        PASD0_INDEX = torch.where(class_out == 0)[0]
        PASD1_INDEX = torch.where(class_out == 1)[0]
        out = torch.zeros((x.shape[0],2,*x.shape[2:5])).to(x.device)
        out[PASD0_INDEX] = self.PASD0([feat[PASD0_INDEX] for feat in mid_feature])
        out[PASD1_INDEX] = self.PASD1([feat[PASD1_INDEX] for feat in mid_feature])

        return out

    def seg_forward(self,x):
        mid_feature = self.msfe(x)
        out0 = self.PASD0(mid_feature)
        out1 = self.PASD1(mid_feature)
        return out0, out1

    def cla_forward(self,x):
        mid_feature = self.msfe(x)
        class_out = self.efc(mid_feature[-1])
        return class_out
