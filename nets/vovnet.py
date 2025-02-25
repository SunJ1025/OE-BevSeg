# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# ------------------------------------------------------------------------
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import warnings

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": False,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch":  [128, 160, 192, 224], # [128, 160, 192, 224]
    "stage_out_ch": [256, 512, 768, 1024],  # [256, 512, 768, 1024]
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],     #  [1, 3, 9, 3]
    "eSE": False,     # True False去掉 ese
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            '{}_{}/dw_conv3x3'.format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
                bias=False
            )
        ),
        (
            '{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        ),
        ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

# from torch.cuda.amp import autocast
# from mamba_ssm import Mamba
# class MambaLayer(nn.Module):
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_bi=False):
#         super().__init__()
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         self.mamba = Mamba(
#             d_model=dim,  # Model dimension d_model
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand,  # Block expansion factor
#             use_bi=use_bi
#         )

#     @autocast(enabled=False)
#     def forward(self, x):
#         if x.dtype == torch.float16:
#             x = x.type(torch.float32)
#         B, C = x.shape[:2]
#         assert C == self.dim
#         n_tokens = x.shape[2:].numel()
#         img_dims = x.shape[2:]
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
#         x_norm = self.norm(x_flat)
#         x_mamba = self.mamba(x_norm)
#         out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

#         return out

class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()
        # self.mamba_layer = MambaLayer(dim=channel)

    def forward(self, x):
        input = x     # [6, 256, 112, 200]
        # import pdb; pdb.set_trace()
        # x = self.mamba_layer(x)
        x = self.avg_pool(x)   # [6, 256, 1, 1]
        x = self.fc(x)         # [6, 256, 1, 1]
        x = self.hsigmoid(x)   # [6, 256, 1, 1]
        return input * x


class _OSA_module(nn.Module):
    def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity  # 是否使用跳跃连接
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch, "{}_reduction".format(module_name), "0"))
            )

        # ---------- 5层 每层一个conv3x3 ---------------- #
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # ------------- feature aggregation --------------- #
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x
        output = []
        output.append(x)  # 先加入输入的feature
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)

        # 遍历5层 加入每层的输出
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        # 在通道维度cat包括输入总共 6 个特征
        x = torch.cat(output, dim=1)
        # 1*1 通道降维 输出维度 concat_ch [128, 256, 512, 768] stage 2 ... stage 5
        xt = self.concat(x)
        # 一个注意力模块
        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
            self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False
    ):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:  # 除了第一个stage 都加入 maxpool 宽和高各变为原来的 1/2
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:  # [1, 3, 9, 3] 实际上包括第一层和最后一层都使用了 SE
            SE = False
        # 先添加一层
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        # 遍历添加层
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise
                ),
            )


class VoVNet(nn.Module):
    def __init__(self, spec_name, input_ch=3, out_features=None,
                 frozen_stages=-1, norm_eval=True, pretrained=None, init_cfg=None):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()
        self.frozen_stages = frozen_stages  # -1
        self.norm_eval = norm_eval  # 冻结 norm

        # 加载与训练权重
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        # 根据模型名字选择参数
        stage_specs = _STAGE_SPECS[spec_name]

        stem_ch = stage_specs["stem"]  # [64, 64, 128]
        config_stage_ch = stage_specs["stage_conv_ch"]  # [128, 160, 192, 224]
        config_concat_ch = stage_specs["stage_out_ch"]  # [256, 512, 768, 1024]
        block_per_stage = stage_specs["block_per_stage"]  # [1, 3, 9, 3]
        layer_per_block = stage_specs["layer_per_block"]  # 5
        SE = stage_specs["eSE"]  # True
        depthwise = stage_specs["dw"]  # False

        # 选择输出的层
        self._out_features = out_features

        # -------------- Stem module -------------------- #
        # 对于输入的 RGB 特征进行预处理  三层卷积 + bn + relu
        conv_type = dw_conv3x3 if depthwise else conv3x3  # 是否使用 dw_conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)  # 3  64  strid 2 长宽各下采样到一半 224, 400
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)  # 64 64  1
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)  # 64 128 2       输出： 112, 200
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))

        current_stirde = 4
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}  # 4 4
        self._out_feature_channels = {"stem": stem_ch[2]}  # 128

        stem_out_ch = [stem_ch[2]]  # 128
        in_ch_list = stem_out_ch + config_concat_ch[:-1]  # [128, 256, 512, 768]
        print(in_ch_list)
        # --------------- OSA stages ------------------- #
        self.stage_names = []
        for i in range(3):  # num_stages  之前是4 现在修改为3 去掉最后一个 stage5 输出没有用到
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],        # [128, 256, 512, 768]
                    config_stage_ch[i],   # [128, 160, 192, 224] 中间层的通道数
                    config_concat_ch[i],  # [256, 512, 768, 1024] 1*1 通道降维 输出维度
                    block_per_stage[i],   # [1, 3, 9, 3]  每个 stage 多少个 block
                    layer_per_block,      # 5 每个 block 多少层
                    i + 2,                # module_name stage_num 2,3,4,5
                    SE,                   # True
                    depthwise,            # False
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

        # initialize weights
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            m = getattr(self, 'stem')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'stage{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed."""
    #     super(VoVNet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()


if __name__ == '__main__':
    from torchinfo import summary

    VoVNet = VoVNet(spec_name='V-99-eSE',
                    input_ch=3,
                    out_features=['stage2', 'stage3', 'stage4'],
                    frozen_stages=-1,
                    norm_eval=True,
                    pretrained=None,
                    init_cfg=None)
    state_dict = torch.load('./vov99_new_weights.pth')
    keys_to_remove = [key for key in state_dict if key.startswith('bbox_head.')]
    keys_to_remove_2 = [key for key in state_dict if key.startswith('stage5.')]
    for key in keys_to_remove:
        del state_dict[key]
    for key in keys_to_remove_2:
        del state_dict[key]
    new_dict = {}
    for key, value in state_dict.items():
        if key.startswith('img_backbone.'):
            new_key = key[13:]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    VoVNet.load_state_dict(new_dict)

    X = torch.randn(2, 3, 448, 800)
    out = VoVNet(X)
    print(VoVNet)
    # print(out['stage3'])
    print(out['stage2'].shape, out['stage3'].shape, out['stage4'].shape)  
    # [2, 256, 112, 200] [2, 512, 56, 100] [2, 768, 28, 50] [2, 1024, 14, 25]
    # print(VoVNet)
    # summary(VoVNet.train(), input_size=(2, 3, 448, 800))  # 假设批大小为1
