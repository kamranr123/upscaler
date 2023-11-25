# code from ComfyUI 
from __future__ import annotations

from collections import OrderedDict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn


class B:
    ####################
    # Basic blocks
    ####################

    @staticmethod
    def act(act_type: str, inplace=True, neg_slope=0.2, n_prelu=1):
        # helper selecting activation
        # neg_slope: for leakyrelu and init of prelu
        # n_prelu: for p_relu num_parameters
        act_type = act_type.lower()
        if act_type == "relu":
            layer = nn.ReLU(inplace)
        elif act_type == "leakyrelu":
            layer = nn.LeakyReLU(neg_slope, inplace)
        elif act_type == "prelu":
            layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
        else:
            raise NotImplementedError(
                "activation layer [{:s}] is not found".format(act_type)
            )
        return layer

    @staticmethod
    def norm(norm_type: str, nc: int):
        # helper selecting normalization layer
        norm_type = norm_type.lower()
        if norm_type == "batch":
            layer = nn.BatchNorm2d(nc, affine=True)
        elif norm_type == "instance":
            layer = nn.InstanceNorm2d(nc, affine=False)
        else:
            raise NotImplementedError(
                "normalization layer [{:s}] is not found".format(norm_type)
            )
        return layer

    @staticmethod
    def pad(pad_type: str, padding):
        # helper selecting padding layer
        # if padding is 'zero', do by conv layers
        pad_type = pad_type.lower()
        if padding == 0:
            return None
        if pad_type == "reflect":
            layer = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            layer = nn.ReplicationPad2d(padding)
        else:
            raise NotImplementedError(
                "padding layer [{:s}] is not implemented".format(pad_type)
            )
        return layer

    @staticmethod
    def get_valid_padding(kernel_size, dilation):
        kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = (kernel_size - 1) // 2
        return padding


    class ConcatBlock(nn.Module):
        # Concat the output of a submodule to its input
        def __init__(self, submodule):
            super(B.ConcatBlock, self).__init__()
            self.sub = submodule

        def forward(self, x):
            output = torch.cat((x, self.sub(x)), dim=1)
            return output

        def __repr__(self):
            tmpstr = "Identity .. \n|"
            modstr = self.sub.__repr__().replace("\n", "\n|")
            tmpstr = tmpstr + modstr
            return tmpstr


    class ShortcutBlock(nn.Module):
        # Elementwise sum the output of a submodule to its input
        def __init__(self, submodule):
            super(B.ShortcutBlock, self).__init__()
            self.sub = submodule

        def forward(self, x):
            output = x + self.sub(x)
            return output

        def __repr__(self):
            tmpstr = "Identity + \n|"
            modstr = self.sub.__repr__().replace("\n", "\n|")
            tmpstr = tmpstr + modstr
            return tmpstr


    class ShortcutBlockSPSR(nn.Module):
        # Elementwise sum the output of a submodule to its input
        def __init__(self, submodule):
            super(B.ShortcutBlockSPSR, self).__init__()
            self.sub = submodule

        def forward(self, x):
            return x, self.sub

        def __repr__(self):
            tmpstr = "Identity + \n|"
            modstr = self.sub.__repr__().replace("\n", "\n|")
            tmpstr = tmpstr + modstr
            return tmpstr

    @staticmethod
    def sequential(*args):
        # Flatten Sequential. It unwraps nn.Sequential.
        if len(args) == 1:
            if isinstance(args[0], OrderedDict):
                raise NotImplementedError("sequential does not support OrderedDict input.")
            return args[0]  # No sequential is needed.
        modules = []
        for module in args:
            if isinstance(module, nn.Sequential):
                for submodule in module.children():
                    modules.append(submodule)
            elif isinstance(module, nn.Module):
                modules.append(module)
        return nn.Sequential(*modules)


    ConvMode = Literal["CNA", "NAC", "CNAC"]


    # 2x2x2 Conv Block
    @staticmethod
    def conv_block_2c2(
        in_nc,
        out_nc,
        act_type="relu",
    ):
        return B.sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=2, padding=1),
            nn.Conv2d(out_nc, out_nc, kernel_size=2, padding=0),
            B.act(act_type) if act_type else None,
        )

    @staticmethod
    def conv_block(
        in_nc: int,
        out_nc: int,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type="zero",
        norm_type: str | None = None,
        act_type: str | None = "relu",
        mode: ConvMode = "CNA",
        c2x2=False,
    ):
        """
        Conv layer with padding, normalization, activation
        mode: CNA --> Conv -> Norm -> Act
            NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
        """

        if c2x2:
            return B.conv_block_2c2(in_nc, out_nc, act_type=act_type)

        assert mode in ("CNA", "NAC", "CNAC"), "Wrong conv mode [{:s}]".format(mode)
        padding = B.get_valid_padding(kernel_size, dilation)
        p = B.pad(pad_type, padding) if pad_type and pad_type != "zero" else None
        padding = padding if pad_type == "zero" else 0

        c = nn.Conv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        a = B.act(act_type) if act_type else None
        if mode in ("CNA", "CNAC"):
            n = B.norm(norm_type, out_nc) if norm_type else None
            return B.sequential(p, c, n, a)
        elif mode == "NAC":
            if norm_type is None and act_type is not None:
                a = B.act(act_type, inplace=False)
                # Important!
                # input----ReLU(inplace)----Conv--+----output
                #        |________________________|
                # inplace ReLU will modify the input, therefore wrong output
            n = B.norm(norm_type, in_nc) if norm_type else None
            return B.sequential(n, a, p, c)
        else:
            assert False, f"Invalid conv mode {mode}"


    ####################
    # Useful blocks
    ####################


    class ResNetBlock(nn.Module):
        """
        ResNet Block, 3-3 style
        with extra residual scaling used in EDSR
        (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
        """

        def __init__(
            self,
            in_nc,
            mid_nc,
            out_nc,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            pad_type="zero",
            norm_type=None,
            act_type="relu",
            mode: B.ConvMode = "CNA",
            res_scale=1,
        ):
            super(B.ResNetBlock, self).__init__()
            conv0 = B.conv_block(
                in_nc,
                mid_nc,
                kernel_size,
                stride,
                dilation,
                groups,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
            )
            if mode == "CNA":
                act_type = None
            if mode == "CNAC":  # Residual path: |-CNAC-|
                act_type = None
                norm_type = None
            conv1 = B.conv_block(
                mid_nc,
                out_nc,
                kernel_size,
                stride,
                dilation,
                groups,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
            )
            # if in_nc != out_nc:
            #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
            #         None, None)
            #     print('Need a projecter in ResNetBlock.')
            # else:
            #     self.project = lambda x:x
            self.res = B.sequential(conv0, conv1)
            self.res_scale = res_scale

        def forward(self, x):
            res = self.res(x).mul(self.res_scale)
            return x + res


    class RRDB(nn.Module):
        """
        Residual in Residual Dense Block
        (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
        """

        def __init__(
            self,
            nf,
            kernel_size=3,
            gc=32,
            stride=1,
            bias: bool = True,
            pad_type="zero",
            norm_type=None,
            act_type="leakyrelu",
            mode: B.ConvMode = "CNA",
            _convtype="Conv2D",
            _spectral_norm=False,
            plus=False,
            c2x2=False,
        ):
            super(B.RRDB, self).__init__()
            self.RDB1 = B.ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                plus=plus,
                c2x2=c2x2,
            )
            self.RDB2 = B.ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                plus=plus,
                c2x2=c2x2,
            )
            self.RDB3 = B.ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                plus=plus,
                c2x2=c2x2,
            )

        def forward(self, x):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
            return out * 0.2 + x


    class ResidualDenseBlock_5C(nn.Module):
        """
        Residual Dense Block
        style: 5 convs
        The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
        Modified options that can be used:
            - "Partial Convolution based Padding" arXiv:1811.11718
            - "Spectral normalization" arXiv:1802.05957
            - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
                {Rakotonirina} and A. {Rasoanaivo}

        Args:
            nf (int): Channel number of intermediate features (num_feat).
            gc (int): Channels for each growth (num_grow_ch: growth channel,
                i.e. intermediate channels).
            convtype (str): the type of convolution to use. Default: 'Conv2D'
            gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
                trainable parameters)
            plus (bool): enable the additional residual paths from ESRGAN+
                (adds trainable parameters)
        """

        def __init__(
            self,
            nf=64,
            kernel_size=3,
            gc=32,
            stride=1,
            bias: bool = True,
            pad_type="zero",
            norm_type=None,
            act_type="leakyrelu",
            mode: B.ConvMode = "CNA",
            plus=False,
            c2x2=False,
        ):
            super(B.ResidualDenseBlock_5C, self).__init__()

            ## +
            self.conv1x1 = B.conv1x1(nf, gc) if plus else None
            ## +

            self.conv1 = B.conv_block(
                nf,
                gc,
                kernel_size,
                stride,
                bias=bias,
                pad_type=pad_type,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                c2x2=c2x2,
            )
            self.conv2 = B.conv_block(
                nf + gc,
                gc,
                kernel_size,
                stride,
                bias=bias,
                pad_type=pad_type,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                c2x2=c2x2,
            )
            self.conv3 = B.conv_block(
                nf + 2 * gc,
                gc,
                kernel_size,
                stride,
                bias=bias,
                pad_type=pad_type,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                c2x2=c2x2,
            )
            self.conv4 = B.conv_block(
                nf + 3 * gc,
                gc,
                kernel_size,
                stride,
                bias=bias,
                pad_type=pad_type,
                norm_type=norm_type,
                act_type=act_type,
                mode=mode,
                c2x2=c2x2,
            )
            if mode == "CNA":
                last_act = None
            else:
                last_act = act_type
            self.conv5 = B.conv_block(
                nf + 4 * gc,
                nf,
                3,
                stride,
                bias=bias,
                pad_type=pad_type,
                norm_type=norm_type,
                act_type=last_act,
                mode=mode,
                c2x2=c2x2,
            )

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(torch.cat((x, x1), 1))
            if self.conv1x1:
                # pylint: disable=not-callable
                x2 = x2 + self.conv1x1(x)  # +
            x3 = self.conv3(torch.cat((x, x1, x2), 1))
            x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
            if self.conv1x1:
                x4 = x4 + x2  # +
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    @staticmethod
    def conv1x1(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    ####################
    # Upsampler
    ####################

    @staticmethod
    def pixelshuffle_block(
        in_nc: int,
        out_nc: int,
        upscale_factor=2,
        kernel_size=3,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type: str | None = None,
        act_type="relu",
    ):
        """
        Pixel shuffle layer
        (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
        Neural Network, CVPR17)
        """
        conv = B.conv_block(
            in_nc,
            out_nc * (upscale_factor**2),
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=None,
            act_type=None,
        )
        pixel_shuffle = nn.PixelShuffle(upscale_factor)

        n = B.norm(norm_type, out_nc) if norm_type else None
        a = B.act(act_type) if act_type else None
        return B.sequential(conv, pixel_shuffle, n, a)

    @staticmethod
    def upconv_block(
        in_nc: int,
        out_nc: int,
        upscale_factor=2,
        kernel_size=3,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type: str | None = None,
        act_type="relu",
        mode="nearest",
        c2x2=False,
    ):
        # Up conv
        # described in https://distill.pub/2016/deconv-checkerboard/
        upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
        conv = B.conv_block(
            in_nc,
            out_nc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            c2x2=c2x2,
        )
        return B.sequential(upsample, conv)


import re
import math
import functools
import torch.nn.functional as F

# Borrowed from https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/archs/ESRGAN.py
# Which enhanced stuff that was already here
class ESRGAN(nn.Module): #RRDBNet
    def __init__(
        self,
        state_dict,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(ESRGAN, self).__init__()
        self.model_arch = "ESRGAN"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch models
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
            # self.model_arch = "RealESRGAN"
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())
        if self.plus:
            self.model_arch = "ESRGAN+"

        self.state = self.new_to_old_arch(self.state)

        self.key_arr = list(self.state.keys())

        self.in_nc: int = self.state[self.key_arr[0]].shape[1]
        self.out_nc: int = self.state[self.key_arr[-1]].shape[0]

        self.scale: int = self.get_scale()
        self.num_filters: int = self.state[self.key_arr[0]].shape[0]

        c2x2 = False
        if self.state["model.0.weight"].shape[-2] == 2:
            c2x2 = True
            self.scale = round(math.sqrt(self.scale / 4))
            self.model_arch = "ESRGAN-2c2"

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        # Detect if pixelunshuffle was used (Real-ESRGAN)
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (
            self.in_nc / 4,
            self.in_nc / 16,
        ):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{self.upsampler}] is not found")

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act,
                c2x2=c2x2,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    act_type=self.act,
                    c2x2=c2x2,
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                            c2x2=c2x2,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        # Adjust these properties for calculations outside of the model
        if self.shuffle_factor:
            self.in_nc //= self.shuffle_factor**2
            self.scale //= self.shuffle_factor

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state):
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        # upconv layers
        max_upconv = 0
        for key in state.keys():
            match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
            if match is not None:
                _, key_num, key_type = match.groups()
                old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
                max_upconv = max(max_upconv, int(key_num) * 3)

        # final layers
        for key in state.keys():
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key in ("conv_last.weight",):
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key in ("conv_last.bias",):
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        # Sort by first numeric value of each layer
        def compare(item1, item2):
            parts1 = item1.split(".")
            parts2 = item2.split(".")
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2

        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

        # Rebuild the output dict in the right order
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

        return out_dict

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            _, _, h, w = x.size()
            mod_pad_h = (
                self.shuffle_factor - h % self.shuffle_factor
            ) % self.shuffle_factor
            mod_pad_w = (
                self.shuffle_factor - w % self.shuffle_factor
            ) % self.shuffle_factor
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
            x = self.model(x)
            return x[:, :, : h * self.scale, : w * self.scale]
        return self.model(x)


class UnsupportedModel(Exception):
    pass


try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception


class UpscaleModel:
    def __init__(self) -> None:
        pass

    def _load_state_dict(self, state_dict):
        state_dict_keys = list(state_dict.keys())

        if "params_ema" in state_dict_keys:
            state_dict = state_dict["params_ema"]
        elif "params-ema" in state_dict_keys:
            state_dict = state_dict["params-ema"]
        elif "params" in state_dict_keys:
            state_dict = state_dict["params"]

        # Regular ESRGAN, "new-arch" ESRGAN, Real-ESRGAN v1
        try:
            model = ESRGAN(state_dict)
        except:
            # pylint: disable=raise-missing-from
            raise UnsupportedModel
        return model

    def _load_torch_file(self, ckpt, safe_load=False, device=None):
        if device is None:
            device = torch.device("cpu")
        if ckpt.lower().endswith(".safetensors"):
            sd = safetensors.torch.load_file(ckpt, device=device.type)
        else:
            if safe_load:
                if not 'weights_only' in torch.load.__code__.co_varnames:
                    print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                    safe_load = False
            if safe_load:
                pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
            else:
                pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
                # pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle) i removed this
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]
            else:
                sd = pl_sd
        return sd

    def _state_dict_prefix_replace(self, state_dict, replace_prefix, filter_keys=False):
        if filter_keys:
            out = {}
        else:
            out = state_dict
        for rp in replace_prefix:
            replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
            for x in replace:
                w = state_dict.pop(x[0])
                out[x[1]] = w
        return out

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f'{model_name} not found!')
            return

        sd = self._load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = self.state_dict_prefix_replace(sd, {"module.":""})
        out = self._load_state_dict(sd).eval()
        return out

    def _get_tiled_scale_steps(self, width, height, tile_x, tile_y, overlap):
        return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

    @torch.inference_mode()
    def _tiled_scale(self, samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, pbar = None):
        output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device="cpu")
        for b in range(samples.shape[0]):
            s = samples[b:b+1]
            out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
            out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
            for y in range(0, s.shape[2], tile_y - overlap):
                for x in range(0, s.shape[3], tile_x - overlap):
                    s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                    ps = function(s_in).cpu()
                    mask = torch.ones_like(ps)
                    feather = round(overlap * upscale_amount)
                    for t in range(feather):
                            mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                    out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                    out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                    if pbar is not None:
                        pbar.update(1)

            output[b:b+1] = out/out_div
        return output

    def upscale_by_model(self, upscale_model, image_tensor):
            device = torch.device(torch.cuda.current_device())
            upscale_model.to(device)
            in_img = image_tensor.movedim(-1,-3).to(device)

            tile = 512
            overlap = 32

            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * self._get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    s = self._tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=None)
                    oom = False
                except OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e

            upscale_model.cpu()
            s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
            return s

    def load_image_as_torch_tensor(self, image_path):
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def convert_PIL_Image_to_torch_tensor(self, images):
        img = torch.zeros((len(images), images[0].size[1], images[0].size[0], 3))
        for i in range(len(images)):
            im = images[i].convert("RGB")
            im = np.array(im).astype(np.float32) / 255.0
            im = torch.from_numpy(im)
            img[i, :, :, :] = im
        return img

    def convert_torch_tensor_to_PIL_Image(self, images):
        img = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img.append(Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)))
        return img

    def save_images(self, images, output_path, image_name):
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(output_path, f"{image_name}.png"), compress_level=4)

    def _lanczos(self, samples, width, height):
        images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
        images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
        images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
        result = torch.stack(images)
        return result

    def _bislerp(self, samples, width, height):
        def slerp(b1, b2, r):
            '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
            
            c = b1.shape[-1]

            #norms
            b1_norms = torch.norm(b1, dim=-1, keepdim=True)
            b2_norms = torch.norm(b2, dim=-1, keepdim=True)

            #normalize
            b1_normalized = b1 / b1_norms
            b2_normalized = b2 / b2_norms

            #zero when norms are zero
            b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
            b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

            #slerp
            dot = (b1_normalized*b2_normalized).sum(1)
            omega = torch.acos(dot)
            so = torch.sin(omega)

            #technically not mathematically correct, but more pleasing?
            res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
            res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

            #edge cases for same or polar opposites
            res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
            res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
            return res
        
        def generate_bilinear_data(length_old, length_new, device):
            coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
            coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
            ratios = coords_1 - coords_1.floor()
            coords_1 = coords_1.to(torch.int64)
            
            coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
            coords_2[:,:,:,-1] -= 1
            coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
            coords_2 = coords_2.to(torch.int64)
            return ratios, coords_1, coords_2

        orig_dtype = samples.dtype
        samples = samples.float()
        n,c,h,w = samples.shape
        h_new, w_new = (height, width)
        
        #linear w
        ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
        coords_1 = coords_1.expand((n, c, h, -1))
        coords_2 = coords_2.expand((n, c, h, -1))
        ratios = ratios.expand((n, 1, h, -1))

        pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
        pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
        ratios = ratios.movedim(1, -1).reshape((-1,1))

        result = slerp(pass_1, pass_2, ratios)
        result = result.reshape(n, h, w_new, c).movedim(-1, 1)

        #linear h
        ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
        coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
        coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
        ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

        pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
        pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
        ratios = ratios.movedim(1, -1).reshape((-1,1))

        result = slerp(pass_1, pass_2, ratios)
        result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
        return result.to(orig_dtype)

    def upscale(self, images, upscale_method, scale_by):
        s = images.movedim(-1,1)
        width = round(s.shape[3] * scale_by)
        height = round(s.shape[2] * scale_by)

        if upscale_method == "bislerp":
            s = self._bislerp(s, width, height)
        elif upscale_method == "lanczos":
            s = self._lanczos(s, width, height)
        else: # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
            s = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)
        return s.movedim(1,-1)
