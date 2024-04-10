import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms as select_query_unique
import numpy as np
from torchvision.ops import deform_conv2d
import math

class LSKblock(nn.Module):
    def __init__(self, dim, hidden_dim, num_repeat, dyfs = False):
        super().__init__()
        # self.convp1 = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1, groups=dim) for _ in range(num_repeat)])
        self.conv0 = nn.ModuleList([nn.Conv2d(dim, dim, 5, padding=2, groups=dim) for _ in range(num_repeat)])
        self.conv_spatial = nn.ModuleList([nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) for _ in range(num_repeat)])
        self.conv1 = nn.ModuleList([nn.Conv2d(dim, dim//2, 1) for _ in range(num_repeat)])
        self.conv2 = nn.ModuleList([nn.Conv2d(dim, dim//2, 1)  for _ in range(num_repeat)])
        self.conv_squeeze = nn.ModuleList([nn.Conv2d(2, 2, 7, padding=3)  for _ in range(num_repeat)])
        self.conv = nn.ModuleList([nn.Conv2d(dim//2, dim, 1)  for _ in range(num_repeat)])

        self.conv_unique_dynamic = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.num_repeat = num_repeat
        self.dyfs = dyfs

        self.w = [nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) if self.dyfs else nn.Identity() for _ in range(num_repeat)]


        # channel attention 
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # # shared MLP
        # self.mlp = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False),
        #     nn.Conv2d(dim, dim, 1, bias=False),
        #     nn.ReLU(inplace=True),
        # )   for _ in range(num_repeat)])
        

    def forward(self, x):

        for i in range(self.num_repeat):  
            # spatial
            # attn0 = self.convp1[i](x)
            # attn1 = self.conv0[i](attn0)
            attn1 = self.conv0[i](x)
            attn2 = self.conv_spatial[i](attn1)

            attn1 = self.conv1[i](attn1)
            attn2 = self.conv2[i](attn2)
            
            attn = torch.cat([attn1, attn2], dim=1)
            avg_attn = torch.mean(attn, dim=1, keepdim=True)
            max_attn, _ = torch.max(attn, dim=1, keepdim=True)
            agg = torch.cat([avg_attn, max_attn], dim=1)
            sig = self.conv_squeeze[i](agg).sigmoid()

            if self.dyfs:
                weight = self.w[i]
                # weight = w / (torch.sum(w, dim=0) + 0.0001)
                attn = weight[0] * attn1 * sig[:,0,:,:].unsqueeze(1) + weight[1] * attn2 * sig[:,1,:,:].unsqueeze(1)
            else:
                attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
            attn = self.conv[i](attn)
            assert not torch.any(torch.isnan(attn))
            x = x * attn + x
        select_fea = self.conv_unique_dynamic(x)
        return self.bn(select_fea)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class CLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = int(2 * hidden_features / 3)
        self.proj = nn.Conv2d(in_features, in_features, 3, 1, 1)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        assert not torch.any(torch.isnan(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, -1, H, W).contiguous()
        return x

class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = nn.Conv2d(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = nn.Conv2d(c1 // 2, self.c, 1, 1, 0)
 
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class DWR(nn.Module):
    def __init__(self, dim, out_dim, act=None) -> None:
        super().__init__()

        self.conv_3x3 = nn.Conv2d(dim, dim // 2, 3, 1, 1)
        
        self.conv_3x3_d1 = nn.Conv2d(dim // 2, dim, 3, 1, 1, 1)
        self.conv_3x3_d3 = nn.Conv2d(dim // 2, dim // 2, 3, 1, 3, 3)
        self.conv_3x3_d5 = nn.Conv2d(dim // 2, dim // 2, 3, 1, 5, 5)
        
        self.conv_1x1 = nn.Conv2d(dim * 2, dim, 1)
        self.conv_t1x1 = nn.Conv2d(dim, out_dim, 1) if out_dim != dim else nn.Identity()
        
    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        x_out = self.conv_t1x1(x_out)
        return x_out

class BiFPNDown(nn.Module):
    def __init__(self, c1, c2, mode='conv'):
        super(BiFPNDown, self).__init__()
        self.conv = nn.Conv2d if mode == 'conv' else GhostModuleV2
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = self.conv(c1, c2, kernel_size=1, stride=1, padding=0) if c1!=c2 else nn.Identity()
        self.silu = nn.SiLU()
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))

class BiFPNup(nn.Module):
    def __init__(self, c1, c2, mode='conv'):
        super(BiFPNup, self).__init__()
        self.conv = nn.Conv2d if mode == 'conv' else GhostModuleV2
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = self.conv(c1, c2, kernel_size=1, stride=1, padding=0) if c1!=c2 else nn.Identity()
        self.silu = nn.SiLU()
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

class DenseBiFPN(nn.Module):
    def __init__(self, c1, c2):
        super(DenseBiFPN, self).__init__()
        self.w = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0) if c1!=c2 else nn.Identity()
        self.silu = nn.SiLU()
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2] + weight[3] * x[3]))

class Simatten(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(Simatten, self).__init__()
        self.convq = nn.Conv2d(in_c, out_c, 1, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, pre_atten, feat):
        # out.shape == feat.shape
        h, w = feat.shape[2:]
        pre_atten = self.bn(self.conv(pre_atten))
        pre_atten = F.adaptive_avg_pool2d(pre_atten, (h,w)) if pre_atten.shape[-1] >= w \
            else F.interpolate(pre_atten, (h,w), mode='bilinear', align_corners=False)
        
        return feat * pre_atten.sigmoid() + pre_atten

class Simcrosatten(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, mode='conv'):
        super(Simcrosatten, self).__init__()
        self.conv = nn.Conv2d if mode == 'conv' else GhostModuleV2
        self.convq = self.conv(in_channels, out_channels // reduction, 1)
        self.convk = self.conv(in_channels, out_channels // reduction, 1)
        self.convv = self.conv(in_channels, in_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, pre_atten, feat):
        b, _, h, w = feat.shape

        pre_atten = F.adaptive_avg_pool2d(pre_atten, (h,w)) if pre_atten.shape[-1] >= w \
            else F.interpolate(pre_atten, (h,w), mode='bilinear', align_corners=False)
        
        proj_query = self.convq(pre_atten).view(b, -1, h * w).permute(0, 2, 1)  # B x N x C''
        proj_key = self.convk(pre_atten).view(b, -1, h * w)  # B x C'' x N

        attention = torch.bmm(proj_query, proj_key).view(b, h * w, h, w)  # B x N x W x H
        attention = (self.avgpool(attention).view(b, -1, h, w)).sigmoid()  # B x 1 x W x H

        proj_value = self.convv(feat)

        out = proj_value * attention  # B x W x H

        return out

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
    self.act = nn.SiLU(inplace=True)

  def forward(self, x):
    offset = self.conv_offset(x)
    out = deform_conv2d(input=x, offset=offset, weight=self.conv.weight, stride=(self.stride,self.stride), padding=(self.padding, self.padding))
    out = self.act(self.batchnorm(out))
    return out

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, ratio=2, dw_size=3, relu=True, mode='attn', args=None, padding = 0, bias=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()
 
        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
 
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),
                                                           mode='nearest')

if __name__ == '__main__':
    # lskbock = ADown(64,128)
    # x1 = torch.randn((1,64,25,25))

    # y = lskbock(x1)
    # print(y.shape)

    # conv = nn.Conv1d(256, 64, 1, bias=False)
    m = Simcrosatten(64, 32, mode='ghost')
    input1 = torch.randn(2, 64, 10,10)
    input2 = torch.randn(2, 64, 40,40)
    y = m(input1, input2)

    print(y.shape)