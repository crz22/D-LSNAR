import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class AdvancedDropout(nn.Module):
    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super(AdvancedDropout, self).__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = nn.Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = nn.Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = nn.Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = nn.Parameter(torch.Tensor([self.init_mu]))

        self.weight_sigma = nn.Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = nn.Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        if self.training:
            #print(input.size())
            if len(input.size()) == 3: #[1,node,c]
                input1 = input.squeeze(dim=0)
            elif len(input.size()) == 4: #[node,c,w,h]
                input1 = torch.mean(input,dim=(2,3))
                #print(input1.shape)
            else:
                print("adrop input size erro")

            b,c = input1.size()
            # parameterized prior
            h = F.linear(input1, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            epsilon = mu + sigma * torch.randn([b,c]).cuda()
            mask = torch.sigmoid(epsilon)

            if len(input.size()) == 3: #[1,node,c]
                mask = mask.unsqueeze(dim=0)
            elif len(input.size()) == 4: #[node,c,w,h]
                mask = mask.view(input.shape[0],input.shape[1],1,1)
            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
            # if out.mean().isnan():
            #     print("wh: ",self.weight_h)
            #     print("input: ",input)
            #     print("h: ", h)
            #     print("mu: ", mu)
            #     print("sigma: ", sigma)
            #     print("mask: ", mask)
            #     print(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
        else:
            #print("ad test")
            out = input

        return out

class GCNLayer(nn.Module):# GCN层
    def __init__(self,input_features,output_features,bias=False,nomal='BN',actf='relu'):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

        if nomal == 'BN':
            self.Norm = nn.BatchNorm1d(output_features,affine=False,track_running_stats=False)#
        elif nomal == 'LN':
            self.Norm = nn.LayerNorm(output_features)#,elementwise_affine=False,bias=False
        else:
            self.Norm = nn.Sequential()

        if actf == 'relu':
            self.actf = nn.ReLU(inplace=True)
        else:
            self.actf = nn.Sequential()

    def reset_parameters(self): #初始化参数
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        #print(x.shape,self.weights.shape)
        support = torch.matmul(x,self.weights)
        #print(support.shape)
        output = torch.matmul(adj,support)
        if self.bias is not None:
            return output+self.bias
        output = self.actf(self.Norm(output))
        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size),  "Embed size needs  to  be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0] # the number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
            #Fills elements of self tensor with value where mask is True

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class CONV2D(nn.Module):
    def __init__(self,input_features,output_features,norm='BN',actf='relu',k=3,s=1,p=1,d=1):
        super(CONV2D, self).__init__()
        model = []
        model += [nn.Conv2d(input_features,output_features,kernel_size=k,stride=s,padding=p,dilation=d)]#padding_mode='circular'
        if norm == 'BN':
            model += [nn.BatchNorm2d(output_features,affine=False,track_running_stats=False)] #

        if actf == 'relu':
            model += [nn.ReLU(inplace=True)]
        elif actf == 'sigmoid':
            model += [nn.Sigmoid()]
        self.layer = nn.Sequential(*model)
    def forward(self,x):
        return self.layer(x)

class spe_feat_extract16(nn.Module):
    def __init__(self,dropout = 0.5):
        super(spe_feat_extract16, self).__init__()
        self.conv1 = CONV2D(9, 32, k=3, s=2, p=1, d=1)
        # self.pool1 = nn.MaxPool2d(2,2)   #[9,16,16]->[32,8,8]
        self.conv2 = CONV2D(32, 32, k=3, s=1, p=1, d=1)
        # self.pool2 = nn.MaxPool2d(2, 2)  # [32,8,8]->[32,8,8]
        self.conv3 = CONV2D(32, 32, k=3, s=2, p=1, d=1)
        # self.pool3 = nn.MaxPool2d(2, 2)  # [32,8,8]->[32,4,4]
        self.conv4 = CONV2D(32, 32, k=3, s=1, p=1, d=2)
        # self.pool4 = nn.MaxPool2d(2, 2)  # [32,4,4]->[32,2,2]
        # self.conv5 = CONV2D(32,32)#,actf='sigmoid'
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y1 = self.dropout(self.conv1(x))
        # y1 = self.pool1(y1)
        #print("y1: ", y1.shape)
        y2 = self.dropout(self.conv2(y1))
        # y2 = self.pool2(y2)
        #print("y2: ", y2.shape)
        y3 = self.dropout(self.conv3(y2))
        # y3 = self.pool3(y3)
        #print("y3: ", y3.shape)
        y4 = self.dropout(self.conv4(y3))
        # y4 = self.pool4(y4)
        #print("y4: ", y4.shape)
        # out= self.conv5(y4)
        out = y4.view(1,x.shape[0], -1)
        #out = torch.softmax(out, dim=1)
        return out

class GCN_RESBLOCK(nn.Module):
    def __init__(self,input_size,output_size,bias=False):
        super(GCN_RESBLOCK, self).__init__()
        self.GCNlayer = GCNLayer(input_size,output_size,bias=bias,nomal=None)
        self.LinearLayer1 = nn.Sequential(
                                         nn.Linear(output_size,output_size),
                                         nn.BatchNorm1d(output_size,affine=False,track_running_stats=False))
        self.LinearLayer2 = nn.Sequential(nn.Linear(input_size,output_size),
                                         nn.BatchNorm1d(output_size,affine=False,track_running_stats=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,adj,x):
        y1 = self.GCNlayer(adj,x)
        y1 = self.LinearLayer1(y1)

        y2 = self.LinearLayer2(x)
        return self.relu(y1+y2)

class self_att_layer(nn.Module):
    def __init__(self, features):
        super(self_att_layer, self).__init__()
        self.k_layer = nn.Linear(features, features, bias=False)
        self.q_layer = nn.Linear(features, features, bias=False)
        self.v_layer = nn.Linear(features, features, bias=False)
        self.input_size = features

    def forward(self, x):
        # print(x.shape)
        k = self.k_layer(x)
        q = self.q_layer(x)
        v = self.v_layer(x)

        att_scores = torch.matmul(q, k.T) / torch.sqrt(torch.tensor(float(self.input_size)))
        att_weight = F.softmax(att_scores, dim=-1)
        output = torch.matmul(att_weight, v)
        # print(att_weight.shape,output.shape,v.shape)
        # if att_weight.mean().isnan():
        #     print(att_weight,output,v)
        return output