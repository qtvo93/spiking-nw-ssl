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

from utils.parameters import Params


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rescaling=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

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


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, x):
        batch_size, seq_length, feature_dim = x.size()
        x = x.view(batch_size, seq_length * feature_dim, 1)

        x = torch.max(x, dim=1).values

        return x


class ACA_DTNET2(pl.LightningModule):
    def __init__(self, mel_input_channels, gcc_input_channels):
        super(ACA_DTNET2, self).__init__()

        self.mel_input_channels = mel_input_channels
        self.gcc_input_channels = gcc_input_channels

        self.ch_rescaling_1 = nn.Sequential(
            nn.Conv2d(
                self.mel_input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
        )
        self.resnet1 = ResBlock(self.mel_input_channels, 64, self.ch_rescaling_1)

        self.ch_rescaling_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.resnet2 = ResBlock(64, 128, self.ch_rescaling_2)

        self.ch_rescaling_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.resnet3 = ResBlock(128, 256, self.ch_rescaling_3)

        self.ch_rescaling_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.resnet4 = ResBlock(256, 512, self.ch_rescaling_4)

        self.ch_rescaling_5 = nn.Sequential(
            nn.Conv2d(
                self.gcc_input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
        )
        self.resnet5 = ResBlock(self.gcc_input_channels, 64, self.ch_rescaling_5)

        self.ch_rescaling_6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.resnet6 = ResBlock(64, 128, self.ch_rescaling_6)

        self.ch_rescaling_7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.resnet7 = ResBlock(128, 256, self.ch_rescaling_7)

        self.ch_rescaling_8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.resnet8 = ResBlock(256, 512, self.ch_rescaling_8)

        self.max_pooling1 = nn.MaxPool2d((5, 4))
        self.max_pooling2 = nn.MaxPool2d((1, 4))
        self.max_pooling3 = nn.MaxPool2d((1, 2))
        self.max_pooling4 = nn.MaxPool2d((1, 1))

        self.dropout = nn.Dropout2d(p=0.05)

        self.stitch = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
            ]
        )

        self.conformer1 = Conformer(
            dim=512,
            depth=2,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=24,
            attn_dropout=0.1,
            ff_dropout=0.05,
            conv_dropout=0.05,
        )
        self.conformer2 = Conformer(
            dim=512,
            depth=2,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=24,
            attn_dropout=0.1,
            ff_dropout=0.05,
            conv_dropout=0.05,
        )

        self.fc = nn.Linear(512, 128)

        self.doa_act = nn.Tanh()
        # self.doa_act = nn.ReLU()
        self.sed_act = nn.Sigmoid()
        self.fc_combined = nn.Linear(512, 1)
        self.fc_singleton = nn.Linear(15, 1)
        # self.fc_singleton = nn.Linear(32, 1)

        # self.adaptive_gain_control = AdaptiveGainControl()
        
        self.fc_leaky = nn.LeakyReLU()  # not used in this version
        self.global_max_pool = (
            GlobalMaxPooling1D()
        )  # Global max pooling layer, not used in this version
        self.dynamic_conv = DynamicConv2d(231, kernel_size=1)

        # self.grl = GradientReversalLayer(alpha=1.0)
        # self.convnet = ConvNet()
        self.future_mapper = DynamicFeatureMapper(target_T=75, target_F=14784)

    def forward(self, x):
        # # x = self.dynamic_conv(x)
        # if x.size(2) != 14784:
        #     import torch.nn.functional as F
        #     x = x.unsqueeze(1)  # Add a channel dimension
        #     x = F.interpolate(x, size=(75, 14784), mode='bilinear', align_corners=False)
        #     x = x.squeeze(1)  # Remove the channel dimension


        # x = self.adaptive_gain_control(x)
        x = self.future_mapper(x)
        x_mel = x[:, :, :self.mel_input_channels*64]
        x_mel = x_mel.contiguous().view(x_mel.size(0), 75, self.mel_input_channels, 64)
        x_mel = x_mel.permute(0, 2, 1, 3)

        x_gcc = x[:, :, self.mel_input_channels*64:]
        x_gcc = x_gcc.contiguous().view(x_gcc.size(0), 75, self.gcc_input_channels, 64)
        x_gcc = x_gcc.permute(0, 2, 1, 3)

        x_mel = self.resnet1(x_mel)
        x_gcc = self.resnet5(x_gcc)

        x_mel = self.max_pooling1(x_mel)
        x_gcc = self.max_pooling1(x_gcc)

        for i in range(3):
            sed_resnet = getattr(self, f"resnet{i+2}")
            doa_resnet = getattr(self, f"resnet{i+6}")

            x_mel = sed_resnet(x_mel)
            x_gcc = doa_resnet(x_gcc)

            stitch_tensor_1 = self.stitch[i][:, 0, 0]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_1.size(0) != x_mel.size(1):
                if stitch_tensor_1.size(0) > x_mel.size(1):
                    stitch_tensor_1 = stitch_tensor_1[: x.size(1)]
                else:
                    stitch_tensor_1 = stitch_tensor_1.repeat(
                        x.size(1) // stitch_tensor_1.size(0)
                    )
            stitch_tensor_2 = self.stitch[i][:, 0, 1]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_2.size(0) != x_gcc.size(1):
                if stitch_tensor_2.size(0) > x_gcc.size(1):
                    stitch_tensor_2 = stitch_tensor_2[: x.size(1)]
                else:
                    stitch_tensor_2 = stitch_tensor_2.repeat(
                        x.size(1) // stitch_tensor_2.size(0)
                    )

            stitch_tensor_3 = self.stitch[i][:, 1, 0]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_3.size(0) != x_mel.size(1):
                if stitch_tensor_3.size(0) > x_mel.size(1):
                    stitch_tensor_3 = stitch_tensor_3[: x.size(1)]
                else:
                    stitch_tensor_3 = stitch_tensor_3.repeat(
                        x.size(1) // stitch_tensor_3.size(0)
                    )

            stitch_tensor_4 = self.stitch[i][:, 1, 1]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_4.size(0) != x_gcc.size(1):
                if stitch_tensor_4.size(0) > x_gcc.size(1):
                    stitch_tensor_4 = stitch_tensor_4[: x.size(1)]
                else:
                    stitch_tensor_4 = stitch_tensor_4.repeat(
                        x.size(1) // stitch_tensor_4.size(0)
                    )

            x_mel = torch.einsum(
                "c, nctf -> nctf", stitch_tensor_1, x_mel
            ) + torch.einsum("c, nctf -> nctf", stitch_tensor_2, x_gcc)
            x_gcc = torch.einsum(
                "c, nctf -> nctf", stitch_tensor_3, x_mel
            ) + torch.einsum("c, nctf -> nctf", stitch_tensor_4, x_gcc)

            max_pooling_method = getattr(self, f"max_pooling{i+2}")
            x_mel = max_pooling_method(x_mel)
            x_gcc = max_pooling_method(x_gcc)

            x_mel = self.dropout(x_mel)
            x_gcc = self.dropout(x_gcc)

        x_mel = self.max_pooling3(x_mel)
        x_gcc = self.max_pooling3(x_gcc)

        x_mel = x_mel.transpose(1, 2).contiguous()
        x_gcc = x_gcc.transpose(1, 2).contiguous()
        x_mel = x_mel.view(x_mel.shape[0], x_mel.shape[1], -1).contiguous()
        x_gcc = x_gcc.view(x_gcc.shape[0], x_gcc.shape[1], -1).contiguous()
        x_mel = self.conformer1(x_mel)
        x_gcc = self.conformer2(x_gcc)
        # x1 = x_mel
        # x2 = x_gcc

        x_mel = torch.einsum(
            "c, ntc -> ntc", self.stitch[3][:, 0, 0], x_mel
        ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 0, 1], x_gcc)
        x_gcc = torch.einsum(
            "c, ntc -> ntc", self.stitch[3][:, 1, 0], x_mel
        ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 1, 1], x_gcc)

        x_mel = self.sed_act(x_mel)
        x_gcc = self.doa_act(x_gcc)

        x_combined = x_mel + x_gcc  # Element-wise addition
        x_combined = self.fc_combined(x_combined)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_combined = self.fc_singleton(x_combined)

        # return x1, x2, x_combined
        return x_combined


class AdaptiveGainControl(nn.Module):
    def __init__(self, target_energy=1, adaptation_rate=0.2):
        super(AdaptiveGainControl, self).__init__()
        self.target_energy = target_energy
        self.adaptation_rate = adaptation_rate

    def forward(self, x):
        energy = torch.mean(x**2, dim=-1, keepdim=True)
        gain = self.target_energy / (energy + 1e-6)
        x = x * gain * self.adaptation_rate + x * (1 - self.adaptation_rate)
        return x


class DynamicConv2d(nn.Module):
    def __init__(
        self, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True
    ):
        super(DynamicConv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x):
        batch_size = x.size(0)
        nb_frames = x.size(1)
        dynamic_number = x.size(2)

        if dynamic_number == 14784:
            return x

        target_size = 14784
        if dynamic_number != target_size:
            weight = nn.Parameter(
                torch.randn(
                    self.out_channels, dynamic_number // Params.n_mels_bins, 1, 1
                )
            ).to(x.device)
            bias = nn.Parameter(torch.randn(self.out_channels)) if self.bias else None

            x_reshaped = x.view(batch_size, nb_frames, Params.n_mels_bins, -1)
            x_reshaped = x_reshaped.permute(0, 3, 1, 2).contiguous()

            out = nn.functional.conv2d(
                x_reshaped,
                weight,
                bias.to(x.device),
                self.stride,
                self.padding,
                self.dilation,
            )

            out = out.permute(0, 2, 3, 1).contiguous()
            out = out.view(batch_size, out.size(1), -1)
            return out


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    

# class ACA_DTNET(pl.LightningModule):
#     def __init__(self, mel_input_channels, gcc_input_channels):
#         super().__init__()
#         self.feature_extractor = ACA_DTNET2(mel_input_channels, gcc_input_channels)
#         # self.classifier = nn.Linear(feature_dim, num_classes)

#         # Domain Classifier
#         self.grl = GradientReversalLayer(alpha=1.0)
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         # class_output = self.classifier(features)
        
#         # Adversarial domain classification
#         reversed_features = self.grl(features)
#         domain_output = self.domain_classifier(reversed_features)

#         return features, domain_output
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(75, 75, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(75, 75, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(75, 75, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(75, 75, kernel_size=3, stride=1, padding=8, dilation=8)
        self.upsample = nn.ConvTranspose1d(75, 75, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.upsample(x)
        return x
    

class DynamicFeatureMapper(nn.Module):
    def __init__(self, target_T=75, target_F=14784):
        super().__init__()
        self.target_T = target_T
        self.target_F = target_F

    def forward(self, x):
        batch, t, f = x.shape
        
        # Step 1: Adjust Time Dimension (T) -> to target_T (75)
        if t != self.target_T:
            x = x.permute(0, 2, 1)  # Change to [batch, F, T] for interpolation
            x = F.interpolate(x, size=self.target_T, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # Back to [batch, target_T, F]
        
        # Step 2: Adjust Feature Dimension (F) -> to target_F (14784)
        if f != self.target_F:
            x = F.adaptive_avg_pool1d(x, self.target_F)

        return x
    
from sklearn.preprocessing import MinMaxScaler

class ACA_DTNET(pl.LightningModule):
    def __init__(self, a, b):
        super(ACA_DTNET, self).__init__()

        self.ch_rescaling_0 = nn.Sequential(
            nn.Conv1d(
                11,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
        )
        self.resnet0 = ResBlock1D(11, 64, self.ch_rescaling_0)
        self.ch_rescaling_1 = nn.Sequential(
            nn.Conv1d(
                10,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
        )
        self.resnet1 = ResBlock1D(10, 64, self.ch_rescaling_1)

        self.ch_rescaling_2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.resnet2 = ResBlock1D(64, 128, self.ch_rescaling_2)

        self.ch_rescaling_3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
        )
        self.resnet3 = ResBlock1D(128, 256, self.ch_rescaling_3)

        self.ch_rescaling_4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
        )
        self.resnet4 = ResBlock1D(256, 512, self.ch_rescaling_4)

        self.max_pooling1 = nn.MaxPool1d(4)
        self.max_pooling2 = nn.MaxPool1d(4)
        self.max_pooling3 = nn.MaxPool1d(4)
        self.max_pooling4 = nn.MaxPool1d(2)

        self.stitch = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
            ]
        )

        self.dropout = nn.Dropout1d(p=0.2)

        self.conformer1 = Conformer(
            dim=512,
            depth=2,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=24,
            attn_dropout=0.05,
            ff_dropout=0.05,
            conv_dropout=0.05,
        )

        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.doa_act = nn.Tanh()
        self.sed_act = nn.Sigmoid()
        self.fc_1 = nn.Linear(128, 1)
        self.final_fc = nn.Linear(16, 1)
        
        self.fc_leaky = nn.LeakyReLU()  # not used in this version
        self.global_max_pool = (
            GlobalMaxPooling1D()
        )  # Global max pooling layer, not used in this version
        self.dynamic_feature_mapper = DynamicFeatureMapper(target_T=21, target_F=1500)
        # self.sinc_conv = SincConv1D(21, 3)
        # self.conv1 = nn.Conv1d(21, 64, 5, stride = 4,dilation=1, padding=1)
        # self.conv2 = nn.Conv1d(64, 128, 4, stride = 3,dilation=2, padding=2)
        # self.conv3 = nn.Conv1d(128, 256, 3, stride = 2,dilation=3, padding=3)
        # self.conv4 = nn.Conv1d(256, 512, 2, stride = 1,dilation=4, padding=4)
        # self.rnn1 = nn.GRU(input_size=21, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2)

        # self.rnn2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True, dropout=0.2)
        
        # self.rnn3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True, dropout=0.2)
      
        # self.rnn4 = nn.GRU(input_size=256, hidden_size=512, num_layers=1, batch_first=True, dropout=0.2)

         # LSTM Layer
        # self.lstm = nn.LSTM(512, 512, num_layers=2, 
        #                     batch_first=True, dropout=0.1)

        # # GRU Layer
        # self.gru = nn.GRU(128, 128, num_layers=2, 
        #                   batch_first=True, dropout=0.1)

        # # Vanilla RNN Layer
        # self.rnn = nn.RNN(128, 128, num_layers=2, 
        #                   batch_first=True, dropout=0.1)

        # Fully connected layer after all RNN and Conformer processing
        self.fc_out = nn.Linear(512, 1)
        self.fc_64 = nn.Linear(64, 1)
        self.FC_LINEAR = nn.Linear(375 , 512)
        self.fc_21_512 = nn.Linear(187, 512)
        self.fc_26 = nn.Linear(26, 1)
        self.fc_combined = nn.Linear(512, 1)
        self.fc_singleton = nn.Linear(2, 1)
        self.fc_5632 = nn.Linear(5632, 512)

    def forward(self, x):
        x_cpu = x.cpu().numpy()
        # also add some randome nosie to the input
        for ch in range(x_cpu.shape[2]):
            scaler = MinMaxScaler()
            scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
            x_cpu[:, :, ch] = scaled_x.T
        # print(x_cpu.shape)

        # Convert back to tensor and move to the original device
        scaled_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

        x = scaled_x.to(x.device).transpose(1, 2).contiguous()
        # x_identity = scaled_x.to(x.device).transpose(1, 2).contiguous()
        # # resnet branch
        # x = self.resnet1(x)
        # x = self.max_pooling1(x)
        # x = self.dropout(x)
        # x = self.resnet2(x)
        # x = self.max_pooling2(x)
        # x = self.dropout(x)
        # # x = self.resnet3(x)
        # # x = self.max_pooling3(x)
        # # x = self.dropout(x)
        # # x = self.resnet4(x)
        # # x = self.max_pooling4(x)
        # # x = self.dropout(x)
        # # x = self.sed_act(x)
        # x = x.transpose(1, 2).contiguous()
        # x = self.conformer1(x)
        # # print(x.shape)
        # x = torch.mean(x, dim=1)
        # x = self.dropout(x)
        # # print(x.shape)
        # x = self.fc_1(x)
        # return x

        # # print(x.shape)

        # # conformer branch
        # x2 = self.max_pooling4(x_identity)
        # x2 = self.fc_21_512(x2)
        # x2 = self.relu(x2)
        # x2 = self.conformer1(x2)
        # x2 = self.dropout(x2)
        # x2 = self.doa_act(x2)
        # # print(x2.shape)
        # x2 = x2.transpose(1, 2).contiguous()
        # # combine the two branches
        # out = torch.cat((x, x2), dim=2)
        # out = self.dropout(out)
        # out = torch.mean(out, dim=1)
        # out = self.fc_26(out)
        
        # return out
        # x = self.resnet1(x)
        # x = self.max_pooling1(x)
        # x = self.dropout(x)
        # x = self.resnet2(x)
        # x = self.max_pooling2(x)
        # x = self.dropout(x)
        # x = self.resnet3(x)
        # x = self.max_pooling3(x)
        # x = self.dropout(x)
        # x = self.resnet4(x)
        # x = self.max_pooling4(x)
        # x = self.dropout(x)
        # x = x.transpose(1, 2).contiguous()
        # x = self.conformer1(x)
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.fc_5632(x)
        # x = self.fc_out(x)

        # return x

        x_mel = x[:, :11, :]

        x_gcc = x[:, 11:, :]

        x_mel = self.resnet0(x_mel)
        x_gcc = self.resnet1(x_gcc)

        x_mel = self.max_pooling1(x_mel)
        x_gcc = self.max_pooling1(x_gcc)

        for i in range(3):
            sed_resnet = getattr(self, f"resnet{i+2}")
            doa_resnet = getattr(self, f"resnet{i+2}")

            x_mel = sed_resnet(x_mel)
            x_gcc = doa_resnet(x_gcc)

            stitch_tensor_1 = self.stitch[i][:, 0, 0]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_1.size(0) != x_mel.size(1):
                if stitch_tensor_1.size(0) > x_mel.size(1):
                    stitch_tensor_1 = stitch_tensor_1[: x.size(1)]
                else:
                    stitch_tensor_1 = stitch_tensor_1.repeat(
                        x.size(1) // stitch_tensor_1.size(0)
                    )
            stitch_tensor_2 = self.stitch[i][:, 0, 1]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_2.size(0) != x_gcc.size(1):
                if stitch_tensor_2.size(0) > x_gcc.size(1):
                    stitch_tensor_2 = stitch_tensor_2[: x.size(1)]
                else:
                    stitch_tensor_2 = stitch_tensor_2.repeat(
                        x.size(1) // stitch_tensor_2.size(0)
                    )

            stitch_tensor_3 = self.stitch[i][:, 1, 0]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_3.size(0) != x_mel.size(1):
                if stitch_tensor_3.size(0) > x_mel.size(1):
                    stitch_tensor_3 = stitch_tensor_3[: x.size(1)]
                else:
                    stitch_tensor_3 = stitch_tensor_3.repeat(
                        x.size(1) // stitch_tensor_3.size(0)
                    )

            stitch_tensor_4 = self.stitch[i][:, 1, 1]
            # Check if dimensions match, adjust if necessary
            if stitch_tensor_4.size(0) != x_gcc.size(1):
                if stitch_tensor_4.size(0) > x_gcc.size(1):
                    stitch_tensor_4 = stitch_tensor_4[: x.size(1)]
                else:
                    stitch_tensor_4 = stitch_tensor_4.repeat(
                        x.size(1) // stitch_tensor_4.size(0)
                    )

            x_mel = torch.einsum(
                "c, nct -> nct", stitch_tensor_1, x_mel
            ) + torch.einsum("c, nct -> nct", stitch_tensor_2, x_gcc)
            x_gcc = torch.einsum(
                "c, nct -> nct", stitch_tensor_3, x_mel
            ) + torch.einsum("c, nct -> nct", stitch_tensor_4, x_gcc)

            max_pooling_method = getattr(self, f"max_pooling{i+2}")
            x_mel = max_pooling_method(x_mel)
            x_gcc = max_pooling_method(x_gcc)

            x_mel = self.dropout(x_mel)
            x_gcc = self.dropout(x_gcc)

        x_mel = self.max_pooling3(x_mel)
        x_gcc = self.max_pooling3(x_gcc)

        x_mel = x_mel.transpose(1, 2).contiguous()
        x_gcc = x_gcc.transpose(1, 2).contiguous()
        # x_mel = x_mel.view(x_mel.shape[0], x_mel.shape[1], -1).contiguous()
        # x_gcc = x_gcc.view(x_gcc.shape[0], x_gcc.shape[1], -1).contiguous()
        x_mel = self.conformer1(x_mel)
        x_gcc = self.conformer1(x_gcc)

        # x_mel = self.dropout(x_mel)
        # x_gcc = self.dropout(x_gcc)

        x_mel = torch.einsum(
            "c, ntc -> ntc", self.stitch[3][:, 0, 0], x_mel
        ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 0, 1], x_gcc)
        x_gcc = torch.einsum(
            "c, ntc -> ntc", self.stitch[3][:, 1, 0], x_mel
        ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 1, 1], x_gcc)

        x_mel = self.sed_act(x_mel)
        x_gcc = self.doa_act(x_gcc)

        x_combined = x_mel + x_gcc  # Element-wise addition
        x_combined - self.dropout(x_combined)
        x_combined = self.fc_combined(x_combined)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_combined - self.dropout(x_combined)
        x_combined = self.fc_singleton(x_combined)

        # return x1, x2, x_combined
        return x_combined


    # def forward(self, x):
    #     # # x = x.transpose(1, 2).contiguous()
    #     # # Min-max normalization to [0, 1]
    #     # min_val = x.min(dim=0, keepdim=True)[0]  # Minimum value per channel
    #     # max_val = x.max(dim=0, keepdim=True)[0]  # Maximum value per channel

    #     # normalized_audio = (x - min_val) / (max_val - min_val)

    #     # # Ensure the output is in the range [0, 1]
    #     # normalized_audio = torch.clamp(normalized_audio, 0, 1)
    #     x_cpu = x.cpu().numpy()
    #     for ch in range(x_cpu.shape[2]):
    #         scaler = MinMaxScaler()
    #         scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
    #         x_cpu[:, :, ch] = scaled_x.T

    #     # Convert back to tensor and move to the original device
    #     scaled_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

    #     # # print("line 530")
    #     # # print(x.shape)
    #     # # x = self.dynamic_feature_mapper(x)

    #     # # x = self.sinc_conv(x)
    #     # # print("line 532")
    #     # # print(x.shape)
    #     x = scaled_x.to(x.device).transpose(1, 2).contiguous()
    #     x = self.resnet1(x)
    #     # x = self.rnn1(x)[0]
    #     # # x = self.conv1(x)
    #     # # print("line 535")
    #     # # print(x.shape)
    #     x = self.max_pooling1(x)
    #     x = self.dropout(x)
    #     # # print("line 538")
    #     # # print(x.shape)
    #     x = self.resnet2(x)
    #     # x = self.rnn2(x)[0]
    #     # # x = self.conv2(x)
    #     # # print("line 541")
    #     # # print(x.shape)
    #     x = self.max_pooling2(x)
    #     x = self.dropout(x)
    #     # # print("line 544")
    #     # # print(x.shape)
    #     x = self.resnet3(x)
    #     # x = self.rnn3(x)[0]
    #     # # x = self.conv3(x)
    #     # # print("line 547")
    #     # # print(x.shape)
    #     x = self.max_pooling3(x)
    #     x = self.dropout(x)
    #     # # print("line 550")
    #     # # print(x.shape)
    #     x = self.resnet4(x)
    #     # x = self.rnn4(x)[0]
    #     # # x = self.conv4(x)
    #     # # print("line 553")
    #     # # print(x.shape)
    #     x = self.max_pooling4(x)
    #     x = self.dropout(x)
    #     # # print("line 556")
    #     # # print(x.shape)
    #     # # x = self.max_pooling1(x)
    #     # x = self.dropout(x)

    #     # # x = x.transpose(1, 2).contiguous()
    #     # # x = x.view(x.shape[0], x.shape[1], -1).contiguous()
  
    #     # x = self.conformer1(x)
    #     # # x = self.conformer1(x)
    #     # # x = self.conformer1(x)
    #     # # x = self.conformer1(x)

    #     # # print("after conformer")
    #     # # print(x.shape)
    #     # # x1 = self.sed_act(x)
    #     # # x2 = self.doa_act(x)
    #     # # x = x1 + x2
    #     # # x = self.max_pooling1(x)
    #     # x = self.relu(x)
    #     # # print(x.shape)
    #     # x = x[:, -1, :]  # Extract the last time-step
    #     # x = self.max_pooling2(x)
    #     # x = self.dropout(x)
    #     # # print(x.shape)
    #     # x = self.fc(x)
    #     # x = self.max_pooling2(x)
    #     # x = self.dropout(x)
    #     # # x = self.fc_1(x)
    #     # # x = x.view(x.shape[0], -1)
    #     # # x = self.max_pooling4(x)
    #     # x = self.final_fc(x)
    #     # x = self.relu(x)

    #     # # print(x.shape)
    #     # x = x.transpose(1, 2).contiguous()
    #     # x, _ = self.lstm(x)

    #     # # Forward pass through GRU
    #     # gru_out, _ = self.gru(lstm_out)

    #     # # Forward pass through Vanilla RNN
    #     # rnn_out, _ = self.rnn(gru_out)

    #     # # Pass through Conformer
    #     # conformer_out = self.conformer1(rnn_out)
        
    #     # x = self.FC_LINEAR(lstm_out)
    #     # x = self.dropout(x)
        
    #     # print(x.shape)
    #     # x = x.view(x.shape[0], x.shape[2], -1).contiguous()
    #     # print(x.shape)

    #     # Use the last hidden state (last time-step output) from the Conformer
    #     conformer_out = self.conformer1(x)
    #     # conformer_out = self.dropout(conformer_out)
    #     # out = conformer_out[:, -1, :]  # Extract the last time-step output
    #     out = torch.mean(conformer_out, dim=1)
    #     out = self.dropout(out)

    #     # Pass through the fully connected layer
    #     out = self.fc_out(out)
    #     out = self.dropout(out)
    #     out = self.fc_64(out)
    #     return out


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, rescaling=None):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
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
    

class SincConv1D(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1):
        super(SincConv1D, self).__init__()
        self.sinc_filter = nn.Parameter(torch.randn(out_channels, 21, kernel_size))
        self.stride = stride
    
    def forward(self, x):
        return F.conv1d(x, self.sinc_filter, stride=self.stride)
