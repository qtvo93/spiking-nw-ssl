# Description: This file contains the inspector for the model training and inference.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-05
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

import os
import torch
import snntorch as snn
import matplotlib.pyplot as plt


class ModelInspector:
    def __init__(self, model, save_dir="./"):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.hooks = []

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, snn.Leaky)):
                self.hooks.append(
                    module.register_forward_hook(self._save_feature_maps(name))
                )
            elif isinstance(
                module, (torch.nn.ReLU, torch.nn.Linear, torch.nn.BatchNorm1d)
            ):
                self.hooks.append(
                    module.register_forward_hook(self._save_activations(name))
                )

    def _save_feature_maps(self, name):
        def hook(module, input, output):
            # shape: (channels, time)
            fmap = output[0].detach().cpu()
            # print(len(fmap))
            # print(fmap.shape)
            max_channels = 64

            if isinstance(module, torch.nn.Conv1d):
                num_channels = fmap.shape[0]
            else:
                num_channels = fmap.shape[1]
            num_channels = min(num_channels, max_channels)
            fmap = fmap[:num_channels]

            n_cols = 4
            n_rows = (num_channels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
            axes = axes.flatten()
            for i in range(n_rows * n_cols):
                ax = axes[i]
                if i < num_channels:
                    ax.plot(fmap[i])
                    ax.set_title(f"Ch {i}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{name}_feature_maps.png")
            plt.close()

        return hook

    def _save_activations(self, name):
        def hook(module, input, output):
            act = output.detach().cpu()
            # torch.save(act, f"{self.save_dir}/{name}_activation.pt")
            plt.figure()
            plt.hist(act.numpy().flatten(), bins=100)
            plt.title(f"Activation Histogram - {name}")
            plt.savefig(f"{self.save_dir}/{name}_activation_hist.png")
            plt.close()

        return hook

    def save_weights(self):
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                w = param.detach().cpu()
                # torch.save(w, f"{self.save_dir}/{name.replace('.', '_')}_weights.pt")
                plt.figure()
                plt.hist(w.numpy().flatten(), bins=100)
                plt.title(f"Weight Histogram - {name}")
                plt.savefig(
                    f"{self.save_dir}/{name.replace('.', '_')}_weights_hist.png"
                )
                plt.close()

    def visualize_attention(self, attn_weights, name):
        num_heads = attn_weights.shape[0]
        height = attn_weights.shape[1]

        # Cap the height of the figure (in inches) to avoid image too large error
        max_fig_height = 16  # Max figure height in inches
        scale = min(max_fig_height / height, 0.01)  # Adjust scale factor
        figsize = (4 * num_heads, height * scale)

        fig, axes = plt.subplots(1, num_heads, figsize=figsize)
        if num_heads == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(attn_weights[i].detach().cpu(), cmap="inferno", aspect="auto")
            ax.set_title(f"Head {i}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{name}_attention.png")
        plt.close()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
