import torch
import torch.nn as  nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import deform_conv2d

import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DyAtten(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyAtten, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float()) # [1.0, 1.0, 0.5, 0.5] 
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float()) # [1.0, 0, 0, 0] 

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1) # BxC
            
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta) # [B, 2k]
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x) # [B, 2*k*channels] where k=2

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v # [B, C, 4]

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.max
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        s, b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(s, b, c)
        y2, _ = self.max_pool(x.view(s, b, c, -1), dim = -1)
        weight = self.w / (torch.sum(self.w, dim=0) + 0.0001)
        y = self.fc(weight[0]*y1 + weight[1]*y2).view(s, b, c, 1, 1)
        return x * y.expand_as(x)

class DeformConv(nn.Module):
  def __init__(self, in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1):
    super(DeformConv, self).__init__()
    self.stride = stride
    self.padding = padding
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.conv_offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
    init_offset = torch.Tensor(np.zeros([2*kernel_size*kernel_size, in_channels, kernel_size, kernel_size]))
    self.conv_offset.weight = torch.nn.Parameter(init_offset)
    self.batchnorm = nn.BatchNorm2d(in_channels)

  def forward(self, x):
    offset = self.conv_offset(x)
    out = deform_conv2d(input=x, offset=offset, weight=self.conv.weight, stride=(self.stride,self.stride), padding=(self.padding, self.padding))
    out = self.batchnorm(out)
    return out

class EncoderLayer(nn.Module):

    def __init__(self, num_level, hidden_dim = 256):
        super().__init__()
        DeformConvLayer = DeformConv(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.deform_conv2d = _get_clones(DeformConvLayer, num_level)

        DownsamplingLayer  = DeformConv(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1)
        self.Downsampling = _get_clones(DownsamplingLayer, num_level - 1)

        UpsamplingLayer = nn.Sequential(
                DeformConv(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.Upsampling = _get_clones(UpsamplingLayer, num_level - 1)
        
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in range(num_level)])

        calayer = CALayer(hidden_dim,reduction=16)
        self.calayers = _get_clones(calayer, 5)

        self.dyrelub = DyAtten(channels=hidden_dim, conv_type='2d')


    def forward(self, feats):
        
        # p0 largest p-1 smallest 
        dcfeats = [self.deform_conv2d[i](feat) for i, feat in enumerate(feats)]

        upfeats = [self.Upsampling[i](feat) for i, feat in enumerate(feats[1:])]
        upfeats.append(None)

        downfeats = [self.Downsampling[i](feat) for i, feat in enumerate(feats[:-1]) ]
        downfeats.insert(0, None)
        
        fusion_feats = [self.calayers[i](torch.stack([j for j in fusion_feat if j is not None])) for i, fusion_feat in enumerate(zip(dcfeats, upfeats, downfeats))]
        outs = [self.dyrelub(self.batchnorms[i](torch.sum(feat,0))) for i, feat in enumerate(fusion_feats)]


        return outs
        
class SimEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_level):
        super().__init__()
        self.encoder_layer = EncoderLayer(num_level, hidden_dim)
        self.layers = _get_clones(self.encoder_layer, num_layers)

        #self.batchnorm = nn.BatchNorm2d(hidden_dim, affine=False)
    def forward(self, feats):
        for layer in self.layers:
            feats = layer(feats)
        return feats
    

if __name__ == '__main__':
    encoder_layer = EncoderLayer(3, 256)
    encoder = SimEncoder(encoder_layer, 2)
    x = [torch.randn((2, 256, 5*(2**i), 5*(2**i))) for i in range(3,0,-1)]
    print([i.shape for i in x])
    out = encoder(x)
    print(out[1].shape)