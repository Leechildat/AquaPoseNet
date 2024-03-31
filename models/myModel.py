import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as checkpoint

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from einops import rearrange, repeat
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)


def channel_shuffle(x, groups):
    """Channel Shuffle operation.
    This function enables cross-group information flow for multiple groups
    convolution layers.
    Args:
        x (Tensor): The input tensor.dkkwkaKk
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.
    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU()
        )

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1, groups=branch_channels),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, inc_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inc_channels),
            nn.ReLU()
        )

        self.expand_conv = nn.Sequential(
            nn.Conv2d(branch_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels)
        )
        self.linear_conv = nn.Sequential(
            nn.Conv2d(mid_channels, branch_channels if stem_channels == self.out_channels else stem_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class IterativeHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    nn.Sequential(
                        nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=3, stride=1, padding=1, groups=self.in_channels[i]),
                        nn.BatchNorm2d(self.in_channels[i]),
                        nn.ReLU(),
                        nn.Conv2d(self.in_channels[i], self.in_channels[i + 1], kernel_size=1),
                        nn.BatchNorm2d(self.in_channels[i + 1]),
                        nn.ReLU()
                    )
                )
            else:
                projects.append(
                    nn.Sequential(
                        nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=3, stride=1, padding=1, groups=self.in_channels[i]),
                        nn.BatchNorm2d(self.in_channels[i]),
                        nn.ReLU(),
                        nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=1),
                        nn.BatchNorm2d(self.in_channels[i]),
                        nn.ReLU()
                    )
                )
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=False,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class LiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            with_cp=True,
            norm_layer=nn.LayerNorm,
            d_state=16,
            attn_drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.with_cp = with_cp

        self.layers = self._make_weighting_blocks(num_blocks, d_state, attn_drop, drop_path, norm_layer)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, d_state, attn_drop, drop_path, norm_layer):
        block_layers = []
        for i in range(self.num_branches):
            # print(self.in_channels[i], num_blocks[i])
            branch_blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=self.in_channels[i],
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for j in range(num_blocks[i])])
            if True:  # is this really applied? Yes, but been overriden later in VSSM!
                def _init_weights(module: nn.Module):
                    for name, p in module.named_parameters():
                        if name in ["out_proj.weight"]:
                            p = p.clone().detach_()  # fake init, just to keep the seed ....
                            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

                self.apply(_init_weights)
            block_layers.append(nn.Sequential(*branch_blocks))

        return nn.Sequential(*block_layers)


    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.Upsample(scale_factor=2**(j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, groups=in_channels[j], bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels[i]),
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, groups=in_channels[j], bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(in_channels[j], in_channels[j], kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.layers[i](x[i].permute(0, 2, 3, 1))
            x[i] = x[i].permute(0, 3, 1, 2)
        out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


class LiteHRNet(nn.Module):
    """Lite-HRNet backbone.
    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_
    https://github.com/HRNet/Lite-HRNet.git
    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    Example:
        # >>> import torch
        # >>> extra = dict(
        # >>>     stage1=dict(
        # >>>         num_modules=1,
        # >>>         num_branches=1,
        # >>>         block='BOTTLENECK',
        # >>>         num_blocks=(4, ),
        # >>>         num_channels=(64, )),
        # >>>     stage2=dict(
        # >>>         num_modules=1,
        # >>>         num_branches=2,
        # >>>         block='BASIC',
        # >>>         num_blocks=(4, 4),
        # >>>         num_channels=(32, 64)),
        # >>>     stage3=dict(
        # >>>         num_modules=4,
        # >>>         num_branches=3,
        # >>>         block='BASIC',
        # >>>         num_blocks=(4, 4, 4),
        # >>>         num_channels=(32, 64, 128)),
        # >>>     stage4=dict(
        # >>>         num_modules=3,
        # >>>         num_branches=4,
        # >>>         block='BASIC',
        # >>>         num_blocks=(4, 4, 4, 4),
        # >>>         num_channels=(32, 64, 128, 256)))
        # >>> self = HRNet(extra, in_channels=1)
        # >>> self.eval()
        # >>> inputs = torch.rand(1, 1, 32, 32)
        # >>> level_outputs = self.forward(inputs)
        # >>> for level_out in level_outputs:
        # ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 norm_eval=False,
                 with_cp=True,
                 ):
        super().__init__()
        self.extra = extra
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'])
        # print(self.stem)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_dims = self.stages_spec['num_dims'][i]
            num_dims = [num_dims[i] for i in range(len(num_dims))]
            # print("####", num_channels_last, num_dims)
            setattr(
                self, f'transition{i}',
                self._make_transition_layer(num_channels_last, num_dims))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_dims, multiscale_output=True)
            setattr(self, f'stage{i}', stage)

        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(in_channels=num_channels_last)
        self.final_layer = nn.Conv2d(
            in_channels=96,     # extra.NUM_DECONV_FILTERS[-1]
            out_channels=11,    # cfg.MODEL.NUM_JOINTS
            kernel_size=1,      # extra.FINAL_CONV_KERNEL
            stride=1,
            padding=1 if 1 == 3 else 0
        )

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_pre_layer[i], kernel_size=3, stride=1, padding=1, groups=num_channels_pre_layer[i], bias=False),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        num_branches = stages_spec['num_branches'][stage_index]
        num_depths = stages_spec['num_depths'][stage_index]
        num_dims = stages_spec['num_dims'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        module_type = stages_spec['module_type'][stage_index]

        modules = []

        for i in range(1):
            modules.append(
                LiteHRModule(
                    num_branches,
                    num_depths,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=True,
                    with_fuse=with_fuse,
                    with_cp=self.with_cp))
                # dim = modules[-1].dim

        return nn.Sequential(*modules), num_dims

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        logger.info('=> init weights from normal distribution')
        if isinstance(pretrained, str):
            pretrained_dict = torch.load(pretrained)
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(f'=> loading {k} pretrained model {pretrained}')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.normal_(m.weight, mean=0, std=0.001)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, bias=0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, val=1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)
        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, f'transition{i}')
            for j in range(self.stages_spec['num_branches'][i]):
                # print(i, self.stages_spec['num_branches'][i])
                # print(i, j)
                # print(i, transition[j])
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])

            # print('@@@', type(x_list[1]))
            y_list = getattr(self, f'stage{i}')[0](x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)
        x = x[0]
        x = self.final_layer(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == "__main__":
    def print_model_structure(model, name="", indent=0):
        indentation = "  " * indent
        print(f"{indentation}{name}: {model.__class__.__name__}")
        for child_name, child_module in model.named_children():
            print_model_structure(child_module, child_name, indent + 1)


    base_dim = 96
    backbone=dict(
        in_channels=3,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_branches=(2, 3, 4),
                num_depths=(
                    (2, 2),
                    (2, 2, 9),
                    (2, 2, 9, 2)
                ),
                module_type=('VSS', 'VSS', 'VSS'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_dims=(
                    (base_dim, base_dim * 2),
                    (base_dim, base_dim * 2, base_dim * 4),
                    (base_dim, base_dim * 2, base_dim * 4, base_dim * 8),
                )),
            with_head=True,
        ))
    # model = LiteHRNet(**backbone).to('cuda')
    #
    # print_model_structure(model)
    #
    # # print(model)
    # image = torch.Tensor(8, 3, 256, 128).to('cuda')    # batch_size最大可以是32
    # outs = model(image)
    # print(outs.shape)
#     # NOTE output shape [2, 40, 120, 120]
#     # import pdb;pdb.set_trace()
