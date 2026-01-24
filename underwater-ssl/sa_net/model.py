# Description: This file contains the core model architecture for the ACA_DTNET model
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from conformer import Conformer

# from utils.parameters import Params
from utils.set_seed import SetSeed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate
from snntorch import spikegen
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class SA_NET(pl.LightningModule):
    def __init__(self):
        super(SA_NET, self).__init__()
        self.set_seed = SetSeed(seed=42)
        self.set_seed.set_seed()

        self.ch_rescaling_1 = nn.Sequential(
            nn.Conv1d(21, 64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
        )
        self.resnet1 = ResBlock1D(21, 64, self.ch_rescaling_1, 7)

        self.ch_rescaling_2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.resnet2 = ResBlock1D(64, 128, self.ch_rescaling_2, 5)

        self.ch_rescaling_3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
        )
        self.resnet3 = ResBlock1D(128, 256, self.ch_rescaling_3, 3)

        self.ch_rescaling_4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
        )
        self.resnet4 = ResBlock1D(256, 512, self.ch_rescaling_4, 3)

        self.max_pooling1 = nn.MaxPool1d(4)
        self.max_pooling2 = nn.MaxPool1d(4)
        self.max_pooling3 = nn.MaxPool1d(4)
        self.max_pooling4 = nn.MaxPool1d(2)
        self.avg_pooling = nn.AvgPool1d(2)  # Experimental average pooling layer

        self.dropout = nn.Dropout1d(p=0.1)

        self.conformer1 = Conformer(
            dim=512,
            depth=2,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=24,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1,
        )

        self.relu = nn.ReLU()
        self.fc_reduced = nn.Linear(512, 1)
        self.fc_singleton = nn.Linear(11, 1)
        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=0.9956, spike_grad=self.spike_grad)
        self.lif2 = snn.Leaky(beta=0.9821, spike_grad=self.spike_grad)
        self.lif3 = snn.Leaky(beta=0.930, spike_grad=self.spike_grad)

    def forward(self, x):
        x_cpu = x.cpu().numpy()
        for ch in range(x_cpu.shape[2]):
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
            x_cpu[:, :, ch] = scaled_x.T

        # Convert back to tensor and move to the original device
        norm_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

        x = norm_x.to(x.device).transpose(1, 2).contiguous()
        mem1 = self.lif1.init_leaky()
        spk_rec = []

        x = self.resnet1(x)
        x = self.max_pooling1(x)
        x = x.permute(2, 0, 1)  # [time_steps, batch_size, channels]
        num_steps = x.size(0)
        # num_steps = 374 experiment
        for t in range(num_steps):
            # x[t] is now [batch_size, channels]
            spk1, mem1 = self.lif1(x[t], mem1)
            spk_rec.append(spk1)

        # Stack preserving batch dimension
        x = torch.stack(spk_rec, dim=0)  # [time_steps, batch_size, channels]
        x = x.permute(1, 2, 0)  # [batch_size, channels, time_steps]
        for i in range(3):
            sed_resnet = getattr(self, f"resnet{i+2}")
            x = sed_resnet(x)
            max_pooling_method = getattr(self, f"max_pooling{i+2}")
            x = max_pooling_method(x)
            x = self.dropout(x)

            if i < 2:
                spk_rec_ls = []
                lif = getattr(self, f"lif{i+2}")
                mem = lif.init_leaky()

                # Permute to put time dimension first
                x_t = x.permute(2, 0, 1)  # [time_steps, batch_size, channels]
                # Get actual time steps
                actual_steps = min(num_steps, x_t.size(0))
                # actual_steps = 64 experiment

                for t in range(actual_steps):
                    spk, mem = lif(x_t[t], mem)
                    spk_rec_ls.append(spk)

                x = torch.stack(spk_rec_ls, dim=0)  # [time_steps, batch_size, channels]
                x = x.permute(1, 2, 0)  # [batch_size, channels, time_steps]

        x = x.transpose(1, 2).contiguous()
        x = self.conformer1(x)
        x = self.fc_reduced(x)
        x = x.view(x.size(0), -1)
        x = self.fc_singleton(x)
        x = self.relu(x)

        return x


class InputDataToSpikingPerceptronLayer(nn.Module):
    pass


class OutputDataToSpikingPerceptronLayer(nn.Module):
    def __init__(self):
        self.reducer = lambda x, dim: x.mean(dim=dim)

    def forward(self, x):
        if type(x) == list:
            x = torch.stack(x)
        return self.reducer(x, 0)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, rescaling=None, kernel_size=3):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.ch_rescaling = rescaling

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.ch_rescaling is not None:
            identity = self.ch_rescaling(identity)

        out += identity
        out = self.relu(out)

        return out
