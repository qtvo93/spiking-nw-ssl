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

n_time_steps, begin_eval = 128, 0
device = "cuda" if torch.cuda.is_available() else "cpu"

class InputDataToSpikingPerceptronLayer(nn.Module):

    def __init__(self, device):
        super(InputDataToSpikingPerceptronLayer, self).__init__()
        self.device = device

        self.reset_state()
        self.to(self.device)

    def reset_state(self):
        #     self.prev_state = torch.zeros([self.n_hidden]).to(self.device)
        pass

    def forward(self, x, is_2D=True):
        x = x.view(x.size(0), -1)  # Flatten 2D image to 1D for FC
        random_activation_perceptron = torch.rand(x.shape).to(self.device)
        return random_activation_perceptron * x


class OutputDataToSpikingPerceptronLayer(nn.Module):

    def __init__(self, average_output=True):
        """
        average_output: might be needed if this is used within a regular neural net as a layer.
        Otherwise, sum may be numerically more stable for gradients with setting average_output=False.
        """
        super(OutputDataToSpikingPerceptronLayer, self).__init__()
        if average_output:
            self.reducer = lambda x, dim: x.sum(dim=dim)
        else:
            self.reducer = lambda x, dim: x.mean(dim=dim)

    def forward(self, x):
        if type(x) == list:
            x = torch.stack(x)
        return self.reducer(x, 0)
        
class SA_NET(pl.LightningModule):
    def __init__(self):
        super(SA_NET, self).__init__()
        self.set_seed = SetSeed(seed=42)
        self.set_seed.set_seed()

        self.ch_rescaling_00 = nn.Sequential(
            nn.Conv1d(
                21,
                64,
                kernel_size=7,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
        )
        self.resnet00 = ResBlock1D(21, 64, self.ch_rescaling_00, 7)

        self.ch_rescaling_01 = nn.Sequential(
            nn.Conv1d(
                22,
                64,
                kernel_size=7,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
        )
        self.resnet01 = ResBlock1D(22, 64, self.ch_rescaling_01, 7)

        self.ch_rescaling_1 = nn.Sequential(
            nn.Conv1d(
                10,
                64,
                kernel_size=7,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
        )
        self.resnet1 = ResBlock1D(10, 64, self.ch_rescaling_1, 7)

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
        self.avg_pooling = nn.AvgPool1d(2)

        self.stitch = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
            ]
        )

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

        # self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        # self.doa_act = nn.Tanh()
        # self.sed_act = nn.Sigmoid()
        # self.fc_1 = nn.Linear(128, 1)
        # self.final_fc = nn.Linear(16, 1)

        # self.fc_leaky = nn.LeakyReLU()  # not used in this version

        # Fully connected layer after all RNN and Conformer processing
        # self.fc_out = nn.Linear(512, 1)
        # self.fc_64 = nn.Linear(64, 1)
        # self.FC_LINEAR = nn.Linear(375, 512)
        # self.fc_21_512 = nn.Linear(187, 512)
        # self.fc_26 = nn.Linear(26, 1)
        self.fc_combined = nn.Linear(512, 1)
        self.fc_singleton = nn.Linear(11, 1)
        # self.fc_singleton = nn.Linear(3, 1)
        
        # self.spiking_net = SpikingNeuralNetwork()
        # self.extra_convo = nn.Conv1d(512, 128, kernel_size=1)

        # self.positional_encoding = PositionalEncoding(embed_dim=256, dropout=0.1, max_len=11)

        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=0.9956, spike_grad=self.spike_grad)
        self.lif2 = snn.Leaky(beta=0.9821, spike_grad=self.spike_grad)
        self.lif3 = snn.Leaky(beta=0.930, spike_grad=self.spike_grad)
        # self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256, dropout=0.2)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # self.fctransansformer = nn.Linear(256, 64)
        # self.fc_singleton2 = nn.Linear(75, 1)
        # self.fc_testing1 = nn.Linear(11,1)
        # self.standardscaler = StandardScaler()
        # self.minmaxscaler = MinMaxScaler()
        # self.save_file = True
        # self.file_count = 1
        self.channel_proj = None
        self.fixed_channels = 21
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1500)

    def forward(self, x):
        # out_dict = {}
        x_cpu = x.cpu().numpy()
        # print("before_Norm", x)
        for ch in range(x_cpu.shape[2]):
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
            x_cpu[:, :, ch] = scaled_x.T

        # Convert back to tensor and move to the original device
        norm_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

        x = norm_x.to(x.device).transpose(1, 2).contiguous()

        # if self.channel_proj is None or x.shape[1] != self.fixed_channels:
        #     # Dynamically create 1x1 conv to match input channels
        #     self.channel_proj = nn.Conv1d(x.shape[1], self.fixed_channels, kernel_size=1).to(x.device)

        # x_mel = self.adaptive_pooling(x)
        # x_mel = self.channel_proj(x_mel

        # print("after Norm", x)
        # x = self.sequence_CNN(x)
        # x = x.transpose(1, 2).contiguous()
        # x = self.conformer1(x)
        # # print(x.shape)
        # # x = self.positional_encoding(x)
        # # # print(x.shape)
        # # x = self.transformer_encoder(x)
        # # # print(x.shape)
        # # # flatten
        # # x = x.view(x.size(0), x.size(1), -1).contiguous()
        # # # print(x.shape)
        # # x = self.fctransansformer(x)
        # # print(x.shape)
        # x = self.spiking_net(x)
        # # print(x.shape)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc_singleton2(x)
        # return x
        # x = x.transpose(1, 2).contiguous()
        mem1 = self.lif1.init_leaky()

        # num_steps = 50
        # num_steps = 100
        spk_rec = []


        # x_mel = x
        # print("after_norm", x_mel)
        # x_mel = x[:, :11, :]

        # # x_gcc = x[:, 11:, :]
        # if x.size(1) == 21:
        #     x_mel = self.resnet00(x)
        # elif x.size(1) == 22:
        #     x_mel = self.resnet01(x)

        # x_gcc = self.resnet1(x_gcc)
        # out_dict["cnn1"] = x_mel.detach().cpu().numpy()
        # print("CNN1", x_mel)
        x_mel = self.resnet00(x)
        x_mel = self.max_pooling1(x_mel) # try pool after steps below
        # x_mel = self.dropout(x_mel)

        # print("Pooling1", x_mel)
        # x_gcc = self.max_pooling1(x_gcc)
        # Change x_mel shape to put time dimension first
        x_mel = x_mel.permute(2, 0, 1)  # [time_steps, batch_size, channels]
        num_steps = x_mel.size(0)  
        # print(num_steps)
        # num_steps = 374
        # print(num_steps)
        # Now iterate over the time dimension
        for t in range(num_steps):
            # x_mel[t] is now [batch_size, channels]
            spk1, mem1 = self.lif1(x_mel[t], mem1)
            spk_rec.append(spk1)

        # Stack preserving batch dimension
        x_mel = torch.stack(spk_rec, dim=0)  # [time_steps, batch_size, channels]
        # print("LIF1", x_mel)
        # out_dict["LIF1"] = x_mel.detach().cpu().numpy()
        # Return to original dimension order if 
        # print("after LIF1", x_mel)
        x_mel = x_mel.permute(1, 2, 0)  # [batch_size, channels, time_steps]
        # print("LIF1", x_mel)
        # x_mel = self.max_pooling1(x_mel) #new co
        # x_mel = self.dropout(x_mel)
        # print(x_mel.shape)
        for i in range(3):
            sed_resnet = getattr(self, f"resnet{i+2}")

            x_mel = sed_resnet(x_mel)
            # out_dict[f"cnn{i+2}"] = x_mel.detach().cpu().numpy()
            # print(f"CNN_{i+2}", x_mel)
            max_pooling_method = getattr(self, f"max_pooling{i+2}")
            x_mel = max_pooling_method(x_mel)
            # print(f"After_Pooling_{i+2}", x_mel)
            x_mel = self.dropout(x_mel)

            if i < 2:
                spk_rec_ls = []
                lif = getattr(self, f"lif{i+2}")
                mem = lif.init_leaky()
                
                # Permute to put time dimension first
                x_mel_t = x_mel.permute(2, 0, 1)  # [time_steps, batch_size, channels]
                
                # Get actual time steps from the data
                actual_steps = min(num_steps, x_mel_t.size(0))
                # actual_steps = x_mel_t.size(0)
                # actual_steps = 64
                
                for t in range(actual_steps):
                    spk, mem = lif(x_mel_t[t], mem)
                    spk_rec_ls.append(spk)

                x_mel = torch.stack(spk_rec_ls, dim=0)  # [time_steps, batch_size, channels]
                # print(f"lif{i+2}", x_mel)
                # out_dict[f"lif{i+2}"] = x_mel.detach().cpu().numpy()
                # Permute back if needed
                x_mel = x_mel.permute(1, 2, 0)  # [batch_size, channels, time_steps]
                # print(f"lif{i+2}", x_mel)
            # x_mel = max_pooling_method(x_mel)
            # x_mel = self.dropout(x_mel)

        # x_mel = self.max_pooling3(x_mel)
  
        x_mel = x_mel.transpose(1, 2).contiguous()
        # # # x_mel = x_mel.view(x_mel.shape[0], x_mel.shape[1], -1).contiguous()
        x_mel = self.conformer1(x_mel)
        # print("Conformer", x_mel)
        # out_dict["conformer"] = x_mel.detach().cpu().numpy()
        # x_mel = self.dropout(x_mel)

        # x_mel = self.sed_act(x_mel)
        # x_mel = self.spiking_net(x_mel)
        # print(x_mel.shape)
        # x_mel = self.extra_convo(x_mel)


        # x_combined = self.dropout(x_combined)
        # x_combined = self.fc_combined(x_combined)
        # # print(x_combined.shape)
       # # x_combined = self.spiking_net(x_combined)
        # print(x_combined.shape)
        # x_combined = self.avg_pooling(x_combined)
        # x_combined = self.spiking_net(x_combined)
       # # x_combined = x_combined.view(x_combined.size(0), -1)
        # x_combined = self.dropout(x_combined)
        
        # x_combined = self.fc_singleton(x_combined)
        x_combined = self.fc_combined(x_mel)
        # print("x_combined_1", x_combined)
        # out_dict["x_combined_1"] = x_combined.detach().cpu().numpy()
        # x_combined = self.fc_testing1(x_mel)
        x_combined = x_combined.view(x_combined.size(0), -1)
        # print("flatten", x_combined)
        # x_combined = self.dropout(x_combined)
        x_combined = self.fc_singleton(x_combined)
        # print("x_combined_2", x_combined)
        # out_dict["x_combined_2"] = x_combined.detach().cpu().numpy()
        x_combined = self.relu(x_combined)
        # x_combined = F.relu(x_combined)
        # print("x_out", x_combined)
        # out_dict["x_out"] = x_combined.detach().cpu().numpy()
        # if self.save_file:
        # with open(f"logs_{self.file_count}.npz", "wb") as f:
        #     np.savez(f, **out_dict)
        #     # self.save_file = False
        #     self.file_count += 1
        return x_combined

class SpikingNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super(SpikingNeuralNetwork, self).__init__()

    # def __init__(self, device, n_time_steps, begin_eval):
    #     super(SpikingNeuralNetwork, self).__init__()
        assert (0 <= begin_eval and begin_eval < n_time_steps)
        # self.device = device
        self.n_time_steps = n_time_steps
        self.begin_eval = begin_eval

        self.input_conversion = InputDataToSpikingPerceptronLayer(device)

        self.layer1 = SpikingNeuronLayer(
            device, n_inputs=5632, n_hidden=100,
            decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5
        )

        self.layer2 = SpikingNeuronLayer(
            device, n_inputs=100, n_hidden=25,
            decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5
        )

        self.output_conversion = OutputDataToSpikingPerceptronLayer(average_output=False)  # Sum on outputs.
        self.prod = 0
        # self.fc_singleton = nn.Linear(75, 1)
        # self.to(device)

    def forward_through_time(self, x):
        """
        This acts as a layer. Its input is non-time-related, and its output too.
        So the time iterations happens inside, and the returned layer is thus
        passed through global average pooling on the time axis before the return
        such as to be able to mix this pipeline with regular backprop layers such
        as the input data and the output data.
        """
        self.input_conversion.reset_state()
        self.layer1.reset_state()
        self.layer2.reset_state()

        out = []

        all_layer1_states = []
        all_layer1_outputs = []
        all_layer2_states = []
        all_layer2_outputs = []
        for _ in range(self.n_time_steps):
            xi = self.input_conversion(x)

            # For layer 1, we take the regular output.
            layer1_state, layer1_output = self.layer1(xi)

            # We take inner state of layer 2 because it's pre-activation and thus acts as out logits.
            layer2_state, layer2_output = self.layer2(layer1_output)

            all_layer1_states.append(layer1_state)
            all_layer1_outputs.append(layer1_output)
            all_layer2_states.append(layer2_state)
            all_layer2_outputs.append(layer2_output)
            out.append(layer2_state)

        out = self.output_conversion(out[self.begin_eval:])
        return out, [[all_layer1_states, all_layer1_outputs], [all_layer2_states, all_layer2_outputs]]

    def forward(self, x):
        out, _ = self.forward_through_time(x)
        # out = self.fc_singleton(out)
        # return F.log_softmax(out, dim=-1)
        # plotting neuron's activations:
        # print(x[0:1].shape)
        if self.prod == 0:
            self.visualize_all_neurons(x[0:1].unsqueeze(1))
            # print("A hidden neuron that looks excited:")
            self.visualize_neuron(x[0:1].unsqueeze(1), layer_idx=0, neuron_idx=0)
            # print("The output neuron of the label:")
            self.visualize_neuron(x[0:1].unsqueeze(1), layer_idx=1, neuron_idx=1)
            self.prod += 1
        # return F.relu(out)
        return out

    def visualize_all_neurons(self, x):
        assert x.shape[0] == 1 and len(x.shape) == 4, (
            "Pass only 1 example to SpikingNeuralNetwork.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)

        for i, (all_layer_states, all_layer_outputs) in enumerate(layers_state):
            layer_state  =  torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
            layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

            self.plot_layer(layer_state, title="Inner state values of neurons for layer {}".format(i))
            self.plot_layer(layer_output, title="Output spikes (activation) values of neurons for layer {}".format(i))

    def visualize_neuron(self, x, layer_idx, neuron_idx):
        assert x.shape[0] == 1 and len(x.shape) == 4, (
            "Pass only 1 example to SpikingNeuralNetwork.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)

        all_layer_states, all_layer_outputs = layers_state[layer_idx]
        layer_state  =  torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
        layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

        self.plot_neuron(layer_state[neuron_idx], title="Inner state values neuron {} of layer {}".format(neuron_idx, layer_idx))
        self.plot_neuron(layer_output[neuron_idx], title="Output spikes (activation) values of neuron {} of layer {}".format(neuron_idx, layer_idx))

    def plot_layer(self, layer_values, title):
        """
        plot the layer
        """
        width = max(16, layer_values.shape[0] / 8)
        height = max(4, layer_values.shape[1] / 8)
        plt.figure(figsize=(width, height))
        plt.imshow(
            layer_values,
            interpolation="nearest",
            cmap=plt.cm.rainbow
        )
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Neurons of layer")
        # plt.show()
        plt.savefig(f"plot_layer-{title}.png")

    def plot_neuron(self, neuron_through_time, title):
        width = max(16, len(neuron_through_time) / 8)
        height = 4
        plt.figure(figsize=(width, height))
        plt.title(title)
        plt.plot(neuron_through_time)
        plt.xlabel("Time")
        plt.ylabel("Neuron's activation")
        # plt.show()
        plt.savefig("plot_neuron.png")

import math
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
# class NonSpikingNeuralNetwork(nn.Module):

#     def __init__(self):
#         super(NonSpikingNeuralNetwork, self).__init__()
#         self.layer1 = nn.Linear(28*28, 100)
#         self.layer2 = nn.Linear(100, 10)

#     def forward(self, x, is_2D=True):
#         x = x.view(x.size(0), -1)  # Flatten 2D image to 1D for FC
#         x = F.relu(self.layer1(x))
#         x = self.layer2(x)
#         return F.log_softmax(x, dim=-1)



# class SincConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
#         super(SincConv1D, self).__init__()
        
#         # Initialize the sinc filters in the weights
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Initialize filter banks (sinc filters)
#         self.filters = nn.Parameter(self.init_sinc_filters(), requires_grad=False)

#     def init_sinc_filters(self):
#         # Initialize sinc filters
#         t = torch.linspace(-np.pi, np.pi, self.kernel_size)
#         sinc_filter = torch.sin(t) / t
#         sinc_filter[torch.isnan(sinc_filter)] = 1.0
#         return sinc_filter.view(1, 1, -1).repeat(self.out_channels, self.in_channels, 1)

#     def forward(self, x):
#         # Convolution using sinc filters
#         return F.conv1d(x, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation)


# class GaussianLowpass(nn.Module):
#     def __init__(self, cutoff_freq, sample_rate, sigma=0.4):
#         super(GaussianLowpass, self).__init__()
#         self.cutoff_freq = cutoff_freq
#         self.sample_rate = sample_rate
#         self.sigma = sigma

#     def forward(self, x):
#         # Gaussian lowpass filter (simplified for demonstration)
#         freq = torch.fft.fftfreq(x.size(-1), d=1/self.sample_rate)
#         gaussian_filter = torch.exp(-0.5 * (freq/self.cutoff_freq)**2)
#         gaussian_filter = gaussian_filter.to(x.device)
#         return torch.fft.irfft(torch.fft.rfft(x, 1) * gaussian_filter, n=x.size(-1), dim=-1)


# class PCENLayer(nn.Module):
#     def __init__(self, alpha=0.98, delta=1e-5, eps=1e-6):
#         super(PCENLayer, self).__init__()
#         self.alpha = alpha
#         self.delta = delta
#         self.eps = eps

#     def forward(self, x):
#         mean = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
#         diff = x - mean
#         return torch.log(torch.abs(diff) + self.eps) + self.delta


# class SincNetPlus(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=51, stride=1, padding=0, sample_rate=16000, name='sincnet_plus'):
#         super(SincNetPlus, self).__init__()
        
#         self.sincconv = SincConv1D(in_channels, out_channels, kernel_size, stride, padding)
#         self.gaussian_lowpass = GaussianLowpass(cutoff_freq=0.5, sample_rate=sample_rate)  # Adjust cutoff_freq as needed
#         self.pcen = PCENLayer()

#         self.activation = nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, x):
#         x = self.sincconv(x)
#         x = self.gaussian_lowpass(x)
#         x = self.pcen(x)
#         x = self.activation(x)
#         return x

# # Input shape
# batch_size, num_channels, num_steps = 32, 21, 1500

# # Spiking layer parameters
# beta = 0.9

# # Poisson spike encoder
# def encode_poisson(input_data):
#     # input_data should be normalized between 0 and 1
#     return spikegen.poisson(datum=input_data, num_steps=num_steps)

# class ACA_DTNET(pl.LightningModule):
#     def __init__(self, mel_input_channels, gcc_input_channels):
#         super(ACA_DTNET, self).__init__()
#         self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
#         self.fc1 = nn.Linear(num_channels, 64)  # reduce dimensionality
#         self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
#         self.fc2 = nn.Linear(64, 128)
#         self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
#         self.fc3 = nn.Linear(128, 256)
        
#         self.surrogate = surrogate.fast_sigmoid(slope=25)
#         # No spikes, leaky integration for regression
#         self.leaky_integrate = snn.Leaky(beta=beta, spike_grad=self.surrogate, init_hidden=True)
#         self.conformer = Conformer(
#             dim=256,
#             depth=2,
#             dim_head=64,
#             heads=8,
#             ff_mult=4,
#             conv_expansion_factor=2,
#             conv_kernel_size=24,
#             attn_dropout=0.05,
#             ff_dropout=0.05,
#             conv_dropout=0.05,
#         )
#         self.fc2 = nn.Linear(64, 1)  # output single regression value

#     def forward(self, x):
#         # x shape: [batch, timesteps, channels]
#         x = x.permute(1, 0, 2)  # -> [timesteps, batch, channels]

#         spk_rec = []
#         # mem = self.lif1.init_leaky()

#         for step in range(num_steps):
#             cur_input = x[step]
#             cur_input = self.fc1(cur_input)        # [batch, hidden_dim]
#             spk = self.lif1(cur_input)            # spike output
#             spk2 = self.fc2(spk)                   # [batch, hidden_dim]
#             spk2 = self.lif2(spk2)                  # spike output
#             spk3 = self.fc3(spk2)                   # [batch, hidden_dim]
#             spk3 = self.lif3(spk3)                  # spike output
#             spk_rec.append(spk3.detach())           # detach to avoid backward-through-graph error

#         spk_stack = torch.stack(spk_rec, dim=0)    # [timesteps, batch, hidden_dim]
        
#         # Optional: Use leaky integration (no spikes) for smoothed features
#         leak_out = self.leaky_integrate(spk_stack)  # [timesteps, batch, hidden_dim]

#         # Conformer expects input: [batch, timesteps, hidden_dim]
#         x_conf = leak_out.permute(1, 0, 2)          # [batch, timesteps, hidden_dim]
#         x_conf = self.conformer(x_conf)            # [batch, timesteps, hidden_dim]

#         # Pool across time (mean pooling)
#         x_pooled = x_conf.mean(dim=1)              # [batch, hidden_dim]

#         output = self.fc2(x_pooled)                # [batch, 1]

#         return output

#     # def forward(self, x):
#     #     # x shape: [batch, timesteps, channels]
#     #     # print(x.shape)
#     #     x = x.permute(1, 0, 2)  # to [timesteps, batch, channels]
#     #     # print(x.shape)
#     #     spk_rec = []
#     #     mem = self.lif1.init_leaky()

#     #     for step in range(num_steps):
#     #         cur_input = x[step]
#     #         cur_input = self.fc1(cur_input)
#     #         spk = self.lif1(cur_input)
#     #         spk_rec.append(spk.detach()) 

#     #     spk_stack = torch.stack(spk_rec, dim=0)  # [timesteps, batch, hidden_dim]
#     #     spk_sum = spk_stack.sum(dim=0)           # [batch, hidden_dim]
    
#     #     # Leaky integration (no spikes) for regression
#     #     leak_out = self.leaky_integrate(spk_sum)  
#     #     output = self.fc2(leak_out)

#     #     return output  # [batch, 1]
    
#         #     self.mel_input_channels = mel_input_channels
#         #     self.gcc_input_channels = gcc_input_channels
#         #     # self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         #     self.num_neurons = 64

#         #     # --- CNN Encoder
#         #     self.encoder = nn.Sequential(
#         #         nn.Conv1d(self.mel_input_channels, 64, kernel_size=9, padding=4),  # (B,64,1500)
#         #         nn.ReLU(),
#         #         nn.MaxPool1d(2),                                           # (B,64,750)
#         #         nn.Conv1d(64, 128, kernel_size=5, padding=2),              # (B,128,750)
#         #         nn.ReLU(),
#         #         nn.AdaptiveAvgPool1d(self.num_neurons),                         # (B,128,num_neurons)
#         #         nn.Conv1d(128, 1, kernel_size=1),                          # (B,1,num_neurons)
#         #     )

#         #     # Art-rSNN parameters
#         #     self.tau = 1.0
#         #     self.A = 0.5
#         #     self.epsilon = 0.01
#         #     self.tau_m = 1.0
#         #     self.beta = 0.9
#         #     self.c = 342.0  # speed of sound in m/s

#         #     self.W = torch.rand(self.num_neurons, self.num_neurons)
#         #     self.V = None
#         #     self.I = None
#         #     self.pos = torch.rand(self.num_neurons) * 5  # initial positions (km)

#         # def forward(self, x, t=0.1):
#         #     """
#         #     x: Tensor of shape (B, 21, 1500)
#         #     """
#         #     x_cpu = x.cpu().numpy()
#         #     # also add some randome nosie to the input
#         #     for ch in range(x_cpu.shape[2]):
#         #         scaler = MinMaxScaler()
#         #         scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
#         #         x_cpu[:, :, ch] = scaled_x.T
#         #     # print(x_cpu.shape)

#         #     # Convert back to tensor and move to the original device
#         #     scaled_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

#         #     x = scaled_x.to(x.device).transpose(1, 2).contiguous()

#         #     self.W = self.W.to(x.device)
#         #     self.pos = self.pos.to(x.device)

#         #     B = x.size(0)
#         #     V = self.encoder(x).squeeze(1)  # (B, num_neurons)
#         #     self.V = V.detach()  # for tracking

#         #     outputs = []
#         #     for b in range(B):
#         #         Vb = V[b]
#         #         Ib = self.update_synaptic_current(Vb)
#         #         Ib = Ib.to(x.device)
#         #         self.I = Ib.detach()

#         #         # Pick max and second max voltages
#         #         i = torch.argmax(Vb)
#         #         Vi = Vb[i]
#         #         Vj, j = torch.topk(Vb, 2)[0][1], torch.topk(Vb, 2)[1][1]

#         #         delta_tsij = t - (t - 0.05)
#         #         kappa = self.estimate_hidden_position(Vi, Vj)
#         #         dij = torch.abs(self.pos[i] - self.pos[j])
#         #         d = self.calculate_distance(Vi, Vj, delta_tsij, dij, kappa)
#         #         new_pos = self.pos[i] + d  # 1D range update
#         #         outputs.append(new_pos)

#         #     return torch.stack(outputs)

#         # def update_synaptic_current(self, Vj):
#         #     delta_ts = 0.05
#         #     delta_ts = torch.tensor(delta_ts, dtype=torch.float32)
#         #     exp_term = torch.exp(-self.c * delta_ts / self.tau)
#         #     return torch.matmul(self.W, Vj * exp_term)

#         # def estimate_hidden_position(self, Vi, Vj):
#         #     return (Vi / (Vj + 1e-6)) + torch.sign(Vj) * self.epsilon

#         # def calculate_distance(self, Vi, Vj, delta_tsij, dij, kappa):
#         #     if torch.abs(Vi - Vj) > 1e-6:
#         #         numerator = self.c * kappa * delta_tsij
#         #         denominator = 1 - kappa + 1e-6
#         #         return (numerator / denominator) * dij**2
#         #     else:
#         #         return dij
#     #     self.spike_grad = surrogate.fast_sigmoid(slope=25)
#     #     self.conv1 = nn.Conv1d(21, 32, kernel_size=4, stride=1, padding=1)
#     #     self.lif1 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, init_hidden=False)
#     #     self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1)
#     #     self.lif2 = snn.Leaky(beta=0.5, spike_grad=self.spike_grad, init_hidden=False)
#     #     self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
#     #     self.lif3 = snn.Leaky(
#     #         beta=0.5, spike_grad=self.spike_grad, init_hidden=False, output=True
#     #     )
#     #     # self.fc = nn.Linear(256, 1)
#     #     self.fc = nn.Linear(672, 1)
#     #     self.maxpool1 = nn.MaxPool1d(2)
#     #     self.maxpool2 = nn.MaxPool1d(2)
#     #     self.maxpool3 = nn.MaxPool1d(2)

#     #     self.conformer = Conformer(
#     #         dim=672,
#     #         depth=2,
#     #         dim_head=64,
#     #         heads=8,
#     #         ff_mult=4,
#     #         conv_expansion_factor=2,
#     #         conv_kernel_size=24,
#     #         attn_dropout=0.05,
#     #         ff_dropout=0.05,
#     #         conv_dropout=0.05,
#     #     )
#     #     self.dropout = nn.Dropout(0.2)
#     #     self.gru = nn.GRU(
#     #         input_size=32 * 30,  # Must match Conformer output dimension
#     #         hidden_size=256,  # Choose appropriate size
#     #         num_layers=1,
#     #         batch_first=True,
#     #         bidirectional=False,
#     #     )
    
#     #     # self.leaf = Leaf(n_filters=21, sample_rate=1500, learnable=True)
#     #     self.sincconv = SincNetPlus(
#     #         out_channels=21, kernel_size=5, sample_rate=1500, in_channels=21
#     #     )

#     # def forward(self, x):
#     #     x_cpu = x.cpu().numpy()

#     #     for ch in range(x_cpu.shape[2]):
#     #         scaler = MinMaxScaler()
#     #         scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
#     #         x_cpu[:, :, ch] = scaled_x.T
#     #     # print(x_cpu.shape)

#     #     # Convert back to tensor and move to the original device
#     #     scaled_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

#     #     x = scaled_x.to(x.device).transpose(1, 2).contiguous()
        
#     #     x = self.sincconv(x)  # (B, C, T)
#     #     # # Optional: Poisson encoding
#     #     # x = SF.poisson(x)
#     #     # x = self.sincconv(x)  # (T, B, C, chunk)
#     #     # Define LEAF
#     #     # leaf = Leaf(n_filters=64, sample_rate=1500, learnable=True)

#     #     # print(x.shape)
#     #     # print()
#     #     # x = x.view(
#     #     #     x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
#     #     # )  # (T, B, C * chunk)
#     #     x = self.dropout(x)

#     #     batch_size, channels, input_size = x.shape
#     #     num_steps = 34
#     #     chunk_size = input_size // num_steps
#     #     x = x.view(batch_size, channels, num_steps, chunk_size).permute(
#     #         2, 0, 1, 3
#     #     )  # (T, B, C, chunk)
#     #     # mem1 = self.lif1.init_leaky()
#     #     # outputs = []
#     #     # for t in range(num_steps):
#     #     #     out = self.conv1(x[t])  # (B, C, chunk) → conv output
#     #     #     spk, mem1 = self.lif1(out, mem1)
#     #     #     spk = spk.view(batch_size, -1)  # flatten if needed
#     #     #     outputs.append(spk)  # (B, F)
#     #     mem1 = self.lif1.init_leaky()
#     #     # mem2 = self.lif2.init_leaky()
#     #     # mem3 = self.lif3.init_leaky()

#     #     outputs = []

#     #     for t in range(num_steps):
#     #         # First conv + spiking layer
#     #         out = self.maxpool1(self.conv1(x[t]))  # (B, C1, H1)
#     #         spk1, mem1 = self.lif1(out, mem1)

#     #         # # Second conv + spiking layer
#     #         # out = self.maxpool2(self.conv2(spk1))  # (B, C2, H2)
#     #         # spk2, mem2 = self.lif2(out, mem2)

#     #         # # Third conv + spiking layer
#     #         # out = self.maxpool3(self.conv3(spk2))  # (B, C3, H3)
#     #         # spk3, mem3 = self.lif3(out, mem3)

#     #         # # Flatten final spike output
#     #         # spk3 = spk3.view(batch_size, -1)
#     #         spk1 = spk1.view(batch_size, -1)
#     #         outputs.append(spk1)

#     #     # Post-process for regression
#     #     out = torch.stack(outputs)  # (T, B, F)
#     #     # out = out.mean(dim=0)             # or reshape/permutation
#     #     # out = self.fc(out)                # fc = nn.Linear(F, 1)
#     #     out = out.permute(1, 0, 2)  # (B, T, F)
#     #     out = self.dropout(out)
#     #     out = self.conformer(out)
#     #     # out, _ = self.gru(out)            # Process sequence with GRU
#     #     # out = out[:, -1, :]
#     #     out = out.mean(dim=1)
#     #     out = self.dropout(out)  # (B, F)
#     #     out = self.fc(out)
#     #     return out  # (B, 1)
#     #     self.mel_input_channels = mel_input_channels
#     #     self.gcc_input_channels = gcc_input_channels

#     #     self.ch_rescaling_0 = nn.Sequential(
#     #         nn.Conv1d(
#     #             21,
#     #             512,
#     #             kernel_size=3,
#     #             stride=1,
#     #             padding=1,
#     #             bias=False,
#     #         ),
#     #         nn.BatchNorm1d(512),
#     #     )
#     #     self.resnet0 = ResBlock1D(21, 512, self.ch_rescaling_0)
#     #     self.ch_rescaling_1 = nn.Sequential(
#     #         nn.Conv1d(
#     #             21,
#     #             64,
#     #             kernel_size=3,
#     #             stride=1,
#     #             padding=1,
#     #             bias=False,
#     #         ),
#     #         nn.BatchNorm1d(64),
#     #     )
#     #     self.resnet1 = ResBlock1D(21, 64, self.ch_rescaling_1)

#     #     self.ch_rescaling_2 = nn.Sequential(
#     #         nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#     #         nn.BatchNorm1d(128),
#     #     )
#     #     self.resnet2 = ResBlock1D(64, 128, self.ch_rescaling_2)

#     #     self.ch_rescaling_3 = nn.Sequential(
#     #         nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
#     #         nn.BatchNorm1d(256),
#     #     )
#     #     self.resnet3 = ResBlock1D(128, 256, self.ch_rescaling_3)

#     #     self.ch_rescaling_4 = nn.Sequential(
#     #         nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
#     #         nn.BatchNorm1d(512),
#     #     )
#     #     self.resnet4 = ResBlock1D(256, 512, self.ch_rescaling_4)

#     #     self.max_pooling1 = nn.MaxPool1d(5)
#     #     self.max_pooling2 = nn.MaxPool1d(4)
#     #     self.max_pooling3 = nn.MaxPool1d(4)
#     #     self.max_pooling4 = nn.MaxPool1d(2)

#     #     self.stitch = nn.ParameterList(
#     #         [
#     #             nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
#     #             nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
#     #             nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
#     #             nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
#     #         ]
#     #     )

#     #     self.dropout = nn.Dropout1d(p=0.2)

#     #     self.conformer1 = Conformer(
#     #         dim=512,
#     #         depth=2,
#     #         dim_head=64,
#     #         heads=8,
#     #         ff_mult=4,
#     #         conv_expansion_factor=2,
#     #         conv_kernel_size=24,
#     #         attn_dropout=0.05,
#     #         ff_dropout=0.05,
#     #         conv_dropout=0.05,
#     #     )

#     #     self.fc = nn.Linear(128, 64)
#     #     self.relu = nn.ReLU()
#     #     self.doa_act = nn.Tanh()
#     #     self.sed_act = nn.Sigmoid()
#     #     self.fc_1 = nn.Linear(128, 1)
#     #     self.final_fc = nn.Linear(16, 1)
#     #     self.fc_combined = nn.Linear(512 , 1)
#     #     self.fc_singleton = nn.Linear(3, 1)

#     #     self.spike_grad = surrogate.fast_sigmoid(slope=25)
#     #     self.lif1 = snn.RLeaky(beta=0.9, conv2d_channels=256, kernel_size=4, spike_grad=self.spike_grad, init_hidden=False, output=True)
#     #     self.conv1 = nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1)
#     #     self.lif2 = snn.RLeaky(beta=0.9, conv2d_channels=256, kernel_size=4, spike_grad=self.spike_grad, init_hidden=False, output=True)
#     #     self.conv2 = nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1)

#     #     self.reg_head = nn.Sequential(
#     #         nn.Linear(512, 256),
#     #         nn.LeakyReLU(),
#     #         nn.Linear(256, 1)
#     #     )


#     # def forward(self, x):
#     #     # x_cpu = x.cpu().numpy()
#     #     # # also add some randome nosie to the input
#     #     # for ch in range(x_cpu.shape[2]):
#     #     #     scaler = MinMaxScaler()
#     #     #     scaled_x = scaler.fit_transform(x_cpu[:, :, ch].T)
#     #     #     x_cpu[:, :, ch] = scaled_x.T
#     #     # # print(x_cpu.shape)

#     #     # # Convert back to tensor and move to the original device
#     #     # scaled_x = torch.tensor(x_cpu, dtype=x.dtype).to(x.device)

#     #     # x = scaled_x.to(x.device).transpose(1, 2).contiguous()
#     #     x = x.transpose(1, 2).contiguous()
#     #     # x_mel = x[:, :11, :]
#     #     # x_mel = x

#     #     # x_gcc = x[:, 20:, :]
#     #     x_gcc = x

#     #     # x_mel = self.resnet0(x_mel)
#     #     x_gcc = self.resnet1(x_gcc)
#     #     # print(x_mel.shape)
#     #     # print(x_gcc.shape)
#     #     # x_mel = self.max_pooling1(x_mel)
#     #     x_gcc = self.max_pooling1(x_gcc)

#     #     for i in range(2):
#     #         # sed_resnet = getattr(self, f"resnet{i+2}")
#     #         doa_resnet = getattr(self, f"resnet{i+2}")

#     #         # x_mel = sed_resnet(x_mel)
#     #         x_gcc = doa_resnet(x_gcc)

#     #         # stitch_tensor_1 = self.stitch[i][:, 0, 0]
#     #         # # Check if dimensions match, adjust if necessary
#     #         # if stitch_tensor_1.size(0) != x_mel.size(1):
#     #         #     if stitch_tensor_1.size(0) > x_mel.size(1):
#     #         #         stitch_tensor_1 = stitch_tensor_1[: x.size(1)]
#     #         #     else:
#     #         #         stitch_tensor_1 = stitch_tensor_1.repeat(
#     #         #             x.size(1) // stitch_tensor_1.size(0)
#     #         #         )
#     #         # stitch_tensor_2 = self.stitch[i][:, 0, 1]
#     #         # # Check if dimensions match, adjust if necessary
#     #         # if stitch_tensor_2.size(0) != x_gcc.size(1):
#     #         #     if stitch_tensor_2.size(0) > x_gcc.size(1):
#     #         #         stitch_tensor_2 = stitch_tensor_2[: x.size(1)]
#     #         #     else:
#     #         #         stitch_tensor_2 = stitch_tensor_2.repeat(
#     #         #             x.size(1) // stitch_tensor_2.size(0)
#     #         #         )

#     #         # stitch_tensor_3 = self.stitch[i][:, 1, 0]
#     #         # # Check if dimensions match, adjust if necessary
#     #         # if stitch_tensor_3.size(0) != x_mel.size(1):
#     #         #     if stitch_tensor_3.size(0) > x_mel.size(1):
#     #         #         stitch_tensor_3 = stitch_tensor_3[: x.size(1)]
#     #         #     else:
#     #         #         stitch_tensor_3 = stitch_tensor_3.repeat(
#     #         #             x.size(1) // stitch_tensor_3.size(0)
#     #         #         )

#     #         # stitch_tensor_4 = self.stitch[i][:, 1, 1]
#     #         # # Check if dimensions match, adjust if necessary
#     #         # if stitch_tensor_4.size(0) != x_gcc.size(1):
#     #         #     if stitch_tensor_4.size(0) > x_gcc.size(1):
#     #         #         stitch_tensor_4 = stitch_tensor_4[: x.size(1)]
#     #         #     else:
#     #         #         stitch_tensor_4 = stitch_tensor_4.repeat(
#     #         #             x.size(1) // stitch_tensor_4.size(0)
#     #         #         )

#     #         # x_mel = torch.einsum(
#     #         #     "c, nct -> nct", stitch_tensor_1, x_mel
#     #         # ) + torch.einsum("c, nct -> nct", stitch_tensor_2, x_gcc)
#     #         # x_gcc = torch.einsum(
#     #         #     "c, nct -> nct", stitch_tensor_3, x_mel
#     #         # ) + torch.einsum("c, nct -> nct", stitch_tensor_4, x_gcc)

#     #         max_pooling_method = getattr(self, f"max_pooling{i+2}")
#     #         # x_mel = max_pooling_method(x_mel)
#     #         x_gcc = max_pooling_method(x_gcc)

#     #         # x_mel = self.dropout(x_mel)
#     #         x_gcc = self.dropout(x_gcc)

#     #     # x_mel = self.max_pooling3(x_mel)
#     #     # x_gcc = self.max_pooling3(x_gcc)

#     #     # batch_size, x_mel_channels, x_mel_input_size = x_mel.shape
#     #     batch_size, x_gcc_channels, x_gcc_input_size = x_gcc.shape
#     #     # print(x_mel_input_size, x_gcc_input_size)
#     #     num_steps = 3
#     #     # x_mel_chunk_size = x_mel_input_size // num_steps
#     #     x_gcc_chunk_size = x_gcc_input_size // num_steps
#     #     # x_mel = x_mel.view(batch_size, x_mel_channels, num_steps, x_mel_chunk_size).permute(
#     #         # 2, 0, 1, 3
#     #     # )  # (T, B, C, chunk)
#     #     x_gcc = x_gcc.view(batch_size, x_gcc_channels, num_steps, x_gcc_chunk_size).permute(
#     #         2, 0, 1, 3
#     #     )
#     #     # mem1 = self.lif1.init_leaky()
#     #     mem2 = self.lif2.init_rleaky()
#     #     # outputs = []
#     #     outputs2 = []
#     #     for t in range(num_steps):
#     #         # out = self.conv1(x_mel[t])  # (B, C, chunk) → conv output
#     #         # spk, mem1 = self.lif1(out, mem1)
#     #         # spk = spk.view(batch_size, -1)  # flatten if needed
#     #         out2 = self.conv2(x_gcc[t])  # (B, C, chunk) → conv output
#     #         spk2, mem2 = self.lif2(out2, mem2)
#     #         spk2 = spk2.view(batch_size, -1)  # flatten if needed
#     #         # outputs.append(spk)  # (B, F)
#     #         outputs2.append(spk2)  # (B, F)

#     #     # x_mel = torch.stack(outputs)  # (T, B, F)
#     #     x_gcc = torch.stack(outputs2)  # (T, B, F)
#     #     # x_mel = x_mel.permute(1, 0, 2)  # (B, T, F)
#     #     x_gcc = x_gcc.permute(1, 0, 2)  # (B, T, F)

#     #     # x_mel = x_mel.transpose(1, 2).contiguous()
#     #     # x_gcc = x_gcc.transpose(1, 2).contiguous()
#     #     # x_mel = x_mel.view(x_mel.shape[0], x_mel.shape[1], -1).contiguous()
#     #     # x_gcc = x_gcc.view(x_gcc.shape[0], x_gcc.shape[1], -1).contiguous()
#     #     # x_mel = self.conformer1(x_mel)
#     #     x_gcc = self.conformer1(x_gcc)

#     #     # x_mel = self.dropout(x_mel)
#     #     # x_gcc = self.dropout(x_gcc)

#     #     # x_mel = torch.einsum(
#     #     #     "c, ntc -> ntc", self.stitch[3][:, 0, 0], x_mel
#     #     # ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 0, 1], x_gcc)
#     #     # x_gcc = torch.einsum(
#     #     #     "c, ntc -> ntc", self.stitch[3][:, 1, 0], x_mel
#     #     # ) + torch.einsum("c, ntc -> ntc", self.stitch[3][:, 1, 1], x_gcc)

#     #     # x_mel = self.sed_act(x_mel)
#     #     # x_gcc = self.doa_act(x_gcc)
#     #     # print(x_gcc.shape)
#     #     # print(x_mel.shape)
#     #     # x_gcc = x_gcc.transpose(1, 2).contiguous()
#     #     # print(x_gcc.shape)
#     #     # print(x_mel.shape)
#     #     # Concatenate along sequence dimension
#     #     # x_combined = torch.cat([x_mel, x_gcc], dim=2)  # Shape: [32, 512, 1503]
#     #     # x_combined = x_mel + x_gcc  # Element-wise addition
#     #     x_combined = self.dropout(x_gcc)
#     #     x_combined = self.fc_combined(x_combined)
#     #     x_combined = x_combined.view(x_combined.size(0), -1)
#     #     x_combined = self.dropout(x_combined)
#     #     x_combined = self.fc_singleton(x_combined)

#     #     # x1: [32, 512, 3]  
#     #     # x2: [32, 512, 1500] 
#     #     # q = x_gcc.permute(0, 2, 1)  # [32, 3, 512]
#     #     # k = x_mel.permute(0, 2, 1)  # [32, 1500, 512]
#     #     # v = k                   # [32, 1500, 512]

#     #     # attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (512 ** 0.5), dim=-1)  # [32, 3, 1500]
#     #     # context = torch.bmm(attn_weights, v)  # [32, 3, 512]
#     #     # context_pooled = context.mean(dim=1)  # [32, 512]
#     #     # out = self.reg_head(context_pooled)   # [32, 1]
#     #     # x_combined = self.reg_head(out)  # [32, 1]

#     #     # return x1, x2, x_combined
#     #     return x_combined
       


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
