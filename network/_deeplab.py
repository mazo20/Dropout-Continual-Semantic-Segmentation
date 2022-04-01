import torch
from torch import nn
from torch.nn import functional as F
import copy

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, opts, aspp_dilate=[12, 24, 36],):
        super(DeepLabHeadV3Plus, self).__init__()
        self.detached_residual = opts.detached_residual
        self.shared_projection = opts.shared_projection
        low_level_out = opts.low_level_out
        final = 256 + low_level_out

        self.aspp = ASPP(in_channels, aspp_dilate, opts)

        if opts.shared_head:
            self.shared_head = nn.Sequential(
                nn.Conv2d(final, opts.num_classif_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(opts.num_classif_features),
                nn.ReLU(inplace=True),
            )
            self.head = nn.ModuleList([
                nn.Conv2d(opts.num_classif_features, c, 1)
                 for c in num_classes
            ])
        else:
            self.shared_head = None
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(final, opts.num_classif_features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(opts.num_classif_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(opts.num_classif_features, c, 1)
                ) for c in num_classes
            ])
        
        #Reduce the resolution of the residual connection with 'atrous' or 'pooling' to match ASPP output
        modules = []
        if opts.proj_dim_reduction == 'atrous':
            modules.append(nn.Conv2d(low_level_channels, low_level_out, 3, stride=4, dilation=4, padding=4, bias=False))
        elif opts.proj_dim_reduction == 'pooling':
            modules.append(nn.AvgPool2d(5, 4))
            modules.append(nn.Conv2d(low_level_channels, low_level_out, 1, bias=False))
        else:
            modules.append(nn.Conv2d(low_level_channels, low_level_out, 1, bias=False))
        
        modules.append(nn.BatchNorm2d(low_level_out))
        modules.append(nn.ReLU(inplace=True))
        module = nn.Sequential(*modules)

        #Copy projection module if separate projections per step
        if self.shared_projection:
            self.project = module
        else:
            self.project = nn.ModuleList([copy.deepcopy(module) for _ in num_classes])

        #If opts.dropout=0, it is just an Identity mapping
        self.drop = nn.Dropout2d(p=opts.dropout) if opts.dropout_type == '2d' else nn.Dropout(p=opts.dropout)

        self._init_weight()

    def forward(self, feature):
        output_feature = self.drop(feature['out'])
        output_feature = self.aspp(output_feature)
        low_level = feature['low_level']
        low_level = low_level.detach() if self.detached_residual else low_level

        if self.shared_projection:
            low_level = self.project(low_level)
            output_feature = F.interpolate(output_feature, size=low_level.shape[2:], mode='bilinear', align_corners=False)
            output_feature = torch.cat([low_level, output_feature], dim=1)
            output_feature = self.shared_head(output_feature) if self.shared_head else output_feature
        
            heads = [h(output_feature) for h in self.head]
        else:
            low_level = [p(low_level) for p in self.project]
            output_feature = F.interpolate(output_feature, size=low_level[0].shape[2:], mode='bilinear', align_corners=False)
            output_features = [torch.cat([low_lvl, output_feature], dim=1) for low_lvl in low_level]

            heads = [self.head[i](output_features[i]) for i in range(len(self.head))]
            
        heads = torch.cat(heads, dim=1)
        return heads
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, opts, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate, opts)

        if opts.shared_head:
            self.shared_head = nn.Sequential(
                AtrousSeparableConvolution(256, opts.num_classif_features, 3, padding=1, bias=False) if opts.separable_head else nn.Conv2d(256, opts.num_classif_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(opts.num_classif_features),
                nn.ReLU(inplace=True),
            )
            self.head = nn.ModuleList([
                nn.Conv2d(opts.num_classif_features, c, 1)
                 for c in num_classes
            ])
        else:
            self.shared_head = None
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, opts.num_classif_features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(opts.num_classif_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(opts.num_classif_features, c, 1)
                ) for c in num_classes
            ])

        #If opts.dropout=0, it is just an Identity mapping
        self.drop = nn.Dropout2d(p=opts.dropout) if opts.dropout_type == '2d' else nn.Dropout(p=opts.dropout)
        
        self._init_weight()

    def forward(self, features):
        output_feature = self.drop(features['out'])
        output_feature = self.aspp(output_feature)
        output_feature = self.shared_head(output_feature) if self.shared_head else output_feature

        heads = [h(output_feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels*4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=4 ),
            # PointWise Conv
            nn.Conv2d( in_channels*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, opts):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(opts.dropout_aspp)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module