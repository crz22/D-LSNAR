from repaire.rep_baselayer import *

class TED_Net(nn.Module):
    def __init__(self,input_size,
                 num_class=2,
                 hidden_size=64,
                 dropout=0.2,
                 max_length = 22,
                 bias=False,
                 device = []):
        super(TED_Net,self).__init__()
        self.input_size=input_size
        self.device = device
        Layer = [hidden_size,hidden_size*2,hidden_size*4,hidden_size*4,hidden_size*2,hidden_size]
        self.feature_embedding = nn.Linear(input_size,Layer[0])
        self.spe_embedding = spe_feat_extract16(dropout=0.2)
        self.position_embedding = nn.Embedding(max_length, Layer[0]+128)
        #self.adrop0 = AdvancedDropout(Layer[0]+128)

        self.T1 = TransformerBlock(Layer[0]+128,heads=4,dropout=0.2,forward_expansion=1)
        self.G1 = GCN_RESBLOCK(Layer[0]+128, Layer[1],bias=bias)
        self.adrop1 = AdvancedDropout(Layer[1])

        self.T2 = TransformerBlock(Layer[1],heads=4,dropout=0.2,forward_expansion=1)
        self.G2 = GCN_RESBLOCK(Layer[1], Layer[2], bias=bias)
        self.adrop2 = AdvancedDropout(Layer[2])

        self.T3 = TransformerBlock(Layer[2], heads=4, dropout=0.2, forward_expansion=1)
        # self.G3 = GCN_RESBLOCK(Layer[2], Layer[3], bias=bias)
        self.adrop3 = AdvancedDropout(Layer[3])
        #
        # self.T4 = TransformerBlock(Layer[3], heads=4, dropout=0.2, forward_expansion=1)
        # self.adrop4 = AdvancedDropout(Layer[3])
        # self.G4 = GCNLayer(Layer[3], Layer[4], bias=bias, nomal=None)

        self.layer_output = nn.Sequential(nn.Linear(Layer[2],num_class))

        self.dropout = nn.Dropout(dropout)

    def forward(self,adj,swc,spe):
        N,C = swc.shape # node_num, channel 3
        swc = swc.unsqueeze(dim=0) #batch 1,
        #adj = adj.unsqueeze(dim=0)
        swc_feat = self.feature_embedding(swc)
        spe_feat = self.spe_embedding(spe)
        #print(swc_feat.shape,spe_feat.shape)
        feat_cat = torch.cat([swc_feat,spe_feat],dim=2)
        pos_feat = self.position_embedding(torch.arange(0, N).expand(1, N).to(self.device))
        #pos_feat = self.position_embedding(pos).to(self.device)

        input_feat = self.dropout(feat_cat+pos_feat)

        y1 = self.T1(input_feat,input_feat,input_feat)
        #print(adj.shape,y1.shape)
        y1 = self.adrop1(self.G1(adj,y1))


        y2 = self.T2(y1,y1,y1)
        y2 = self.adrop2(self.G2(adj,y2))

        y3 = self.T3(y2,y2,y2)
        y3 = self.adrop3(y3)#self.G3(adj,y3)
        #
        # y4 = self.T4(y3,y3,y3)
        # y4 = self.G4(adj,y4)
        #print(y3.shape)
        out = torch.mean(y3,dim=1,keepdim=True)
        #print(out.shape)
        out = self.layer_output(out)
        #print("x: ", x[0])
        return out.squeeze(dim=0)

class Encoder_SPE(nn.Module):
    def __init__(self, block='res', att=False, a_dp=False):
        super().__init__()
        self.att = att
        self.a_dp = a_dp
        self.layer_input = nn.Sequential(nn.Linear(3, 64),
                                         nn.BatchNorm1d(64, affine=False, track_running_stats=False),
                                         nn.ReLU())
        self.layer_sep = spe_feat_extract16()

        self.layers_config = [
            (192, 32), (32, 48), (48, 64),
            (64, 80), (80, 96), (96, 112), (112, 128), (128, 144)
        ]
        gcn_class = GCN_RESBLOCK if block == 'res' else GCNLayer

        self.gcn_layers = nn.ModuleList()
        self.att = nn.ModuleList()
        self.dp = nn.ModuleList()
        for in_ch, out_ch in self.layers_config[:-1]:
            self.gcn_layers.append(gcn_class(in_ch, out_ch))
            self.att.append(self_att_layer(out_ch))
            self.dp.append(AdvancedDropout(out_ch))
            # self.dp.append(nn.Dropout(0.5))

        self.out_layer = gcn_class(self.layers_config[-1][0], self.layers_config[-1][1])#, actf='none'

    def forward(self, feat_xyz, adj, feat_sp):
        # print(feat_sp.shape)
        swc_feat0 = self.layer_input(feat_xyz)
        spe_feat0 = self.layer_sep(feat_sp)
        spe_feat0 = spe_feat0.squeeze(dim=0)
        x = torch.cat((swc_feat0, spe_feat0), dim=1)
        # print(x.shape)
        for i, (gcn, att, dp) in enumerate(zip(self.gcn_layers, self.att, self.dp)):
            x = gcn(adj, x)
            if self.att and self.a_dp:
                x_a = att(x)
                x = dp(x + x_a)

        x = self.out_layer(adj, x)
        return x

class Decoder(nn.Module):
    def forward(self, z):
        adj = torch.mm(z, z.transpose(1, 0))
        return adj

class TLP_Net(nn.Module):
    def __init__(self, encode='SPE', block='res'):
        super(TLP_Net,self).__init__()
        if encode == 'SPE':
            self.encode = Encoder_SPE(block=block, att=True, a_dp=True)
        self.decode = Decoder()

    def forward(self, feat_xyz, adj, feat_sp):
        z = self.encode(feat_xyz, adj, feat_sp)
        reconstruction = self.decode(z)
        return reconstruction
