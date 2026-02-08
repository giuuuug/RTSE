"""
    ERBTCNN: Deriva da STFT-TCNN ma con compressione/espansione ERB:
    input [B, 257, T] -> ERB analysis [B, N, T] -> TCNN -> ERB synthesis [B, 257, T] -> activation
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                 causal=False, layer_activation="relu"):
        super(DepthwiseSeparableConv, self).__init__()
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = dilation

        if layer_activation == "prelu":
            act = nn.PReLU()
        elif layer_activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError("layer_activation must be one of ['relu', 'prelu']")

        depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=False
        )

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(
                depthwise_conv,
                Chomp1d(padding),
                nn.BatchNorm1d(in_channels),
                act,
                pointwise_conv
            )
        else:
            self.net = nn.Sequential(
                depthwise_conv,
                nn.BatchNorm1d(in_channels),
                act,
                pointwise_conv
            )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, layer_activation="relu"):
        super(ResBlock, self).__init__()
        if layer_activation == "prelu":
            act = nn.PReLU(num_parameters=1)
        elif layer_activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError("layer_activation must be one of ['relu', 'prelu']")

        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            act,
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                causal=False,
                layer_activation=layer_activation
            )
        )

    def forward(self, input):
        x = self.TCM_net(input)
        return x + input


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6,
                 layer_activation="relu"):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i
            layers += [
                ResBlock(
                    in_channels, out_channels,
                    kernel_size, dilation=dilation_size,
                    layer_activation=layer_activation
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ERB(nn.Module):
    """
    Improved ERB with proper normalization for stable reconstruction.

    For STFT-TCNN tensors:
      x: (B, F, T)  where F = n_fft//2 + 1

    bm(x) -> (B, F_low + N, T)
    bs(z) -> (B, F, T)

    Notes:
    - The low band [0:erb_subband_1] is passed through unchanged.
    - The high band [erb_subband_1:F] is projected to N ERB bands using properly normalized filters.
    - Synthesis uses normalized transpose with energy preservation.
    """
    def __init__(self, erb_subband_1: int, erb_subband_2: int, nfft: int = 512, high_lim: int = 8000, fs: int = 16000, learnable: bool = False):
        super().__init__()
        self.erb_subband_1 = int(erb_subband_1)
        self.erb_subband_2 = int(erb_subband_2)
        self.learnable = learnable

        nfreqs = nfft // 2 + 1
        assert 0 <= self.erb_subband_1 < nfreqs, "erb_subband_1 out of range"
        assert self.erb_subband_2 > 0, "erb_subband_2 must be > 0"

        erb_filters = self.erb_filter_banks(self.erb_subband_1, self.erb_subband_2, nfft, high_lim, fs)

        row_sums = erb_filters.sum(dim=1, keepdim=True).clamp_min(1e-8)  # (N, 1)
        erb_filters_norm = erb_filters / row_sums

        col_sums = erb_filters.sum(dim=0, keepdim=True).clamp_min(1e-8)  # (1, F_high)
        erb_filters_col_norm = erb_filters / col_sums

        self.erb_fc = nn.Linear(nfreqs - self.erb_subband_1, self.erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(self.erb_subband_2, nfreqs - self.erb_subband_1, bias=False)

        self.erb_fc.weight = nn.Parameter(erb_filters_norm, requires_grad=learnable)

        syn_filters = erb_filters_col_norm.T
        self.ierb_fc.weight = nn.Parameter(syn_filters, requires_grad=learnable)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1.0)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1.0) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        nfreqs = nfft // 2 + 1

        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)

        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)

        erb_filters = np.zeros([erb_subband_2, nfreqs], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)

        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1.0 - erb_filters[-2, bins[-2]:bins[-1] + 1]

        erb_filters = erb_filters[:, erb_subband_1:]
        erb_filters = np.abs(erb_filters).astype(np.float32)

        return torch.from_numpy(erb_filters)

    def bm(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x[:, :self.erb_subband_1, :]
        x_high = x[:, self.erb_subband_1:, :]

        xh = x_high.transpose(1, 2)
        yh = self.erb_fc(xh)
        yh = yh.transpose(1, 2)

        return torch.cat([x_low, yh], dim=1)

    def bs(self, x_erb: torch.Tensor) -> torch.Tensor:
        x_low = x_erb[:, :self.erb_subband_1, :]
        x_high_erb = x_erb[:, self.erb_subband_1:, :]

        yh = x_high_erb.transpose(1, 2)
        xh = self.ierb_fc(yh)
        xh = xh.transpose(1, 2)

        return torch.cat([x_low, xh], dim=1)


class FullBandGate(nn.Module):
    """
    Full-band conditioning lightweight:
    produce a time-dependent gate g(B,C,T) in (0,1) and modulate features: x <- x * g.

    Uses only 1x1 Conv1d + ReLU + 1x1 Conv1d + Sigmoid.
    Embedded/ONNX friendly.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        hidden = max(1, channels // reduction)

        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        return self.net(x)


class ERB_TCNN(nn.Module):
    """
    ERB-TCNN: copia della STFTTCNN, ma con ERB analysis/synthesis.

    Pipeline:
      input magnitude (B, 257, T)
        -> ERB analysis      (B, C_erb, T)   where C_erb = subband_1 + n_bands
        -> (NEW) Full-band gate (B, C_erb, T) and modulation
        -> TCNN blocks       (B, C_erb, T)
        -> ERB synthesis     (B, 257, T)
        -> activation        (B, 257, T)
    """
    def __init__(self,
                 in_channels=257,
                 tcn_latent_dim=512,
                 n_blocks=3,
                 kernel_size=3,
                 num_layers=6,
                 mask_activation="tanh",
                 layer_activation="relu",
                 init_dilation=2,
                 # ERB params
                 n_fft: int = 512,
                 sample_rate: int = 16000,
                 n_bands: int = 64,
                 subband_1: int = 64,
                 f_max: float = 8000.0,
                 # NEW: full-band conditioning
                 use_fullband_gate: bool = True,
                 fb_gate_reduction: int = 8,
                 **kwargs):
        super().__init__()

        if in_channels != (n_fft // 2 + 1):
            raise ValueError(f"in_channels ({in_channels}) must be n_fft//2+1 ({n_fft//2+1})")

        self.in_channels = in_channels
        self.tcn_latent_dim = tcn_latent_dim
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.mask_activation = mask_activation
        self.layer_activation = layer_activation
        self.init_dilation = init_dilation
        self.subband_1 = subband_1
        self.n_bands = n_bands

        self.register_buffer("identity_mask_bias", torch.ones(1))


        self.erb = ERB(
            erb_subband_1=self.subband_1,
            erb_subband_2=self.n_bands,
            nfft=n_fft,
            high_lim=int(f_max),
            fs=sample_rate,
            learnable=True
        )

        self.erb_channels = self.subband_1 + self.n_bands

        # NEW: full-band gate
        self.use_fullband_gate = bool(use_fullband_gate)
        if self.use_fullband_gate:
            self.fb_gate = FullBandGate(self.erb_channels, reduction=fb_gate_reduction)
        else:
            self.fb_gate = None

        self.tcn_block_dict = OrderedDict()
        for k in range(n_blocks):
            self.tcn_block_dict[f"tcn_block_{k}"] = TCNN_Block(
                in_channels=self.erb_channels,
                out_channels=self.tcn_latent_dim,
                kernel_size=self.kernel_size,
                init_dilation=self.init_dilation,
                num_layers=self.num_layers,
                layer_activation=self.layer_activation
            )

        self.tcn = nn.Sequential(self.tcn_block_dict)

        if self.mask_activation == "tanh":
            self.activation = nn.Tanh()
        elif self.mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("mask_activation must be one of ['tanh', 'sigmoid']")

    def forward(self, input):
        # input: (B, 257, T)
        x_erb = self.erb.bm(input)          # (B, C_erb, T)

        # NEW: full-band conditioning (lightweight gating)
        if self.use_fullband_gate:
            gate = self.fb_gate(x_erb)      # (B, C_erb, T) in (0,1)
            x_erb = x_erb * gate

        tcn_out = self.tcn(x_erb)           # (B, C_erb, T)
        residual = self.erb.bs(tcn_out)           # (B,257,T)
        residual = residual + self.identity_mask_bias
        out = self.activation(residual)
        return out