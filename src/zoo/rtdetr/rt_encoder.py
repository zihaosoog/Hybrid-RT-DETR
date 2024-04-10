'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register
from .my_module import *
from .base_encoder import *
from .dysample import DySample
__all__ = ['BiEncoder', 'PBiEncoder', 'PGBiEncoder', 'HyEncoder']


@register
class PGBiEncoder(BaseEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 # new set
                 spatial_select=False,
                 layer_focus_num=3,):
        super(PGBiEncoder, self).__init__(in_channels = in_channels,
                 feat_strides = feat_strides,
                 hidden_dim = hidden_dim,
                 nhead = nhead,
                 dim_feedforward = dim_feedforward,
                 use_encoder_idx = use_encoder_idx,
                 num_encoder_layers = num_encoder_layers,
                 expansion = expansion,
                 depth_mult = depth_mult,
                 eval_spatial_size = eval_spatial_size,
                 # new set
                 spatial_select = spatial_select,
                 layer_focus_num = layer_focus_num)
        
        self.pan_layer = PBiPAN(in_channels = in_channels,
                 hidden_dim = hidden_dim,
                 expansion = expansion,
                 depth_mult = depth_mult,
                 act = act,
                 mode = 'ghost')

    def basencoder(self, proj_feats):
        return self.pan_layer(proj_feats)


@register
class PBiEncoder(BaseEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 # new set
                 spatial_select=False,
                 layer_focus_num=3,
                 dyfs = False):
        super(PBiEncoder, self).__init__(in_channels = in_channels,
                 feat_strides = feat_strides,
                 hidden_dim = hidden_dim,
                 nhead = nhead,
                 dim_feedforward = dim_feedforward,
                 use_encoder_idx = use_encoder_idx,
                 num_encoder_layers = num_encoder_layers,
                 expansion = expansion,
                 depth_mult = depth_mult,
                 eval_spatial_size = eval_spatial_size,
                 # new set
                 spatial_select = spatial_select,
                 layer_focus_num = layer_focus_num,
                 dyfs = dyfs)
        
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.bifdown = nn.ModuleList()
        self.simcosatn = nn.ModuleList()

        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(DeformConv(hidden_dim, hidden_dim, 3, 1))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.bifdown.append(BiFPNup(hidden_dim, hidden_dim))
            self.simcosatn.append(Simcrosatten(hidden_dim, hidden_dim))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.bifup = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                DeformConv(hidden_dim, hidden_dim, 3, 2)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.bifup.append(BiFPNup(hidden_dim, hidden_dim))

        

    def basencoder(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # global and low cross attention
            global_feat_low = self.simcosatn[len(self.in_channels)-1-idx](proj_feats[-1], feat_low)

            feat = self.bifdown[len(self.in_channels)-1-idx]([upsample_feat, feat_low, global_feat_low])
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](feat)
            assert not torch.any(torch.isnan(inner_out))
            inner_outs.insert(0, inner_out)
            del feat, global_feat_low, upsample_feat, feat_heigh, feat_low

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            feat = self.bifup[idx]([downsample_feat, feat_height, proj_feats[idx + 1]])
            out = self.pan_blocks[idx](feat)
            assert not torch.any(torch.isnan(out))
            outs.append(out)

            del feat, feat_height, feat_low, out, downsample_feat
        del inner_outs, proj_feats

        return outs

   
@register
class BiEncoder(BaseEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 # new set
                 spatial_select=False,
                 layer_focus_num=3,):
        super(BiEncoder, self).__init__(in_channels = in_channels,
                 feat_strides = feat_strides,
                 hidden_dim = hidden_dim,
                 nhead = nhead,
                 dim_feedforward = dim_feedforward,
                 use_encoder_idx = use_encoder_idx,
                 num_encoder_layers = num_encoder_layers,
                 expansion = expansion,
                 depth_mult = depth_mult,
                 eval_spatial_size = eval_spatial_size,
                 # new set
                 spatial_select = spatial_select,
                 layer_focus_num = layer_focus_num)
        
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.bifdown = nn.ModuleList()
        self.simcosatn = nn.ModuleList()

        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.bifdown.append(BiFPNup(hidden_dim, hidden_dim))
            self.simcosatn.append(Simcrosatten(hidden_dim, hidden_dim))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.bifup = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.bifup.append(BiFPNup(hidden_dim, hidden_dim))

        

    def basencoder(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # global and low cross attention
            global_feat_low = self.simcosatn[len(self.in_channels)-1-idx](proj_feats[-1], feat_low)

            feat = self.bifdown[len(self.in_channels)-1-idx]([upsample_feat, feat_low, global_feat_low])
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](feat)
            assert not torch.any(torch.isnan(inner_out))
            inner_outs.insert(0, inner_out)
            del feat, global_feat_low, upsample_feat, feat_heigh, feat_low

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            feat = self.bifup[idx]([downsample_feat, feat_height, proj_feats[idx + 1]])
            out = self.pan_blocks[idx](feat)
            assert not torch.any(torch.isnan(out))
            outs.append(out)
            del feat
        return outs


@register
class HyEncoder(BaseEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 # new set
                 spatial_select=False,
                 layer_focus_num=3,):
        super(HyEncoder, self).__init__(in_channels = in_channels,
                 feat_strides = feat_strides,
                 hidden_dim = hidden_dim,
                 nhead = nhead,
                 dim_feedforward = dim_feedforward,
                 use_encoder_idx = use_encoder_idx,
                 num_encoder_layers = num_encoder_layers,
                 expansion = expansion,
                 depth_mult = depth_mult,
                 eval_spatial_size = eval_spatial_size,
                 # new set
                 spatial_select = spatial_select,
                 layer_focus_num = layer_focus_num)
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        

    def basencoder(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        return outs


class PBiPAN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 hidden_dim=256,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 mode = 'conv'):
        super(PBiPAN, self).__init__()
        self.in_channels = in_channels
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.bifdown = nn.ModuleList()
        self.simcosatn = nn.ModuleList()

        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(DeformConv(hidden_dim, hidden_dim, 3, 1))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, mode=mode)
            )
            self.bifdown.append(BiFPNup(hidden_dim, hidden_dim, mode=mode))
            self.simcosatn.append(Simcrosatten(hidden_dim, hidden_dim, mode=mode))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.bifup = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                DeformConv(hidden_dim, hidden_dim, 3, 2)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, mode=mode)
            )
            self.bifup.append(BiFPNup(hidden_dim, hidden_dim, mode=mode))

        

    def forward(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # global and low cross attention
            global_feat_low = self.simcosatn[len(self.in_channels)-1-idx](proj_feats[-1], feat_low)

            feat = self.bifdown[len(self.in_channels)-1-idx]([upsample_feat, feat_low, global_feat_low])
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](feat)
            assert not torch.any(torch.isnan(inner_out))
            inner_outs.insert(0, inner_out)
            del feat, global_feat_low, upsample_feat, feat_heigh, feat_low

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            feat = self.bifup[idx]([downsample_feat, feat_height, proj_feats[idx + 1]])
            out = self.pan_blocks[idx](feat)
            assert not torch.any(torch.isnan(out))
            outs.append(out)
            del feat
        return outs
