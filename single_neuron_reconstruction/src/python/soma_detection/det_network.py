from soma_detection.det_baselayer import *

class ClassNet(nn.Module):
    def __init__(self, num_classes=2,k_size=7):
        super(ClassNet, self).__init__()
        #[1,256,256]
        self.conv1 = nn.Sequential(CONV2D_BLOCK(1,8,'BN','LeakyReLU', k_size=k_size),
                                   nn.MaxPool2d(2,2),)
        #[8,128,128]
        self.conv2 = nn.Sequential(CONV2D_BLOCK(8,16,'BN','LeakyReLU',k_size=k_size),
                                   nn.MaxPool2d(2,2),)
        #[16,64,64]
        self.conv3 = nn.Sequential(CONV2D_BLOCK(16,32,'BN','LeakyReLU',k_size=k_size),
                                   nn.MaxPool2d(2,2),)
        # [32,32,32]
        self.conv4 = nn.Sequential(CONV2D_BLOCK(32,64,'BN','LeakyReLU',k_size=k_size),
                                   nn.MaxPool2d(2,2),)
        # [64,16,16]
        self.conv5 = nn.Sequential(CONV2D_BLOCK(64,128,'BN','LeakyReLU',k_size=k_size),
                                   nn.MaxPool2d(2,2),)
        # [128,8,8]
        self.conv6 = nn.Sequential(CONV2D_BLOCK(128,256,'BN','LeakyReLU',k_size=3),
                                   nn.MaxPool2d(2,2),)
        # [256,4,4]
        self.conv7 = nn.Sequential(CONV2D_BLOCK(256,512,'BN','LeakyReLU',k_size=3),
                                   nn.MaxPool2d(2,2),)
        # [512,2,2]
        self.fcn1 = nn.Sequential(nn.Linear(2048, 512),
                                  nn.LeakyReLU(0.2,inplace=True),)
        self.fcn6 = nn.Linear(512, num_classes)
    def forward(self, x):
        # print('x', x.shape)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c7_out = c7.reshape(c7.size(0), -1)
        # print(c7.shape,c7_out.shape)
        fc1 = self.fcn1(c7_out)
        fc6 = self.fcn6(fc1)
        return fc6


class SFSNet(nn.Module):
    def __init__(self,normal = 'BN'):
        super(SFSNet, self).__init__()
        # [1,64,128,128]
        self.down_layer1 = nn.Sequential(CONV3D_BLOCK(1,32,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(32,64,normal,activation='LeakyReLU'))
        self.pool1 = nn.MaxPool3d(2)
        # [64,32,64,64]
        self.down_layer2 = nn.Sequential(CONV3D_BLOCK(64,64,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(64,128,normal,activation='LeakyReLU'))
        self.pool2 = nn.MaxPool3d(2)
        # [128,16,32,32]
        self.down_layer3 = nn.Sequential(CONV3D_BLOCK(128,128,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(128,256,normal,activation='LeakyReLU'))
        self.pool3 = nn.MaxPool3d(2)
        # [256,8,16,16]
        self.down_layer4 = nn.Sequential(CONV3D_BLOCK(256,256,normal,activation='LeakyReLU'),
                                   CONV3D_BLOCK(256,512,normal,activation='LeakyReLU'))
        self.sapf = SAPF(512)
        # [512,8,16,16]
        self.up1 = UPSAMPLE(512,mode='CT')
        self.up_layer1 = nn.Sequential(CONV3D_BLOCK(512+256,256,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(256,256,normal,activation='LeakyReLU'))
        # [256,16,32,32]
        self.up2 = UPSAMPLE(256,mode='CT')
        self.up_layer2 = nn.Sequential(CONV3D_BLOCK(256+128,128,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(128,128,normal,activation='LeakyReLU'))
        #[128,32,64,64]
        self.up3 = UPSAMPLE(128,mode='CT')
        self.up_layer3 = nn.Sequential(CONV3D_BLOCK(128+64,64,normal,activation='LeakyReLU'),
                                       CONV3D_BLOCK(64,64,normal,activation='LeakyReLU'))
        # [64,64,128,128]
        self.outlayer = CONV3D_BLOCK(64,2)


    def forward(self, x):
        y1 = self.down_layer1(x)
        d1 = self.pool1(y1)
        y2 = self.down_layer2(d1)
        d2 = self.pool2(y2)
        y3 = self.down_layer3(d2)
        d3 = self.pool3(y3)
        y4 = self.down_layer4(d3)

        y4 = self.sapf(y4)

        u1 = self.up1(y4)
        z1 = self.up_layer1(torch.cat([u1,y3],1))
        u2 = self.up2(z1)
        z2 = self.up_layer2(torch.cat([u2,y2],1))
        u3 = self.up3(z2)
        z3 = self.up_layer3(torch.cat([u3,y1],1))

        out = self.outlayer(z3)
        return out

