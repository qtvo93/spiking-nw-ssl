# Description: This file contains the Inference class which is responsible for performing inference on the test dataset
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Sequence

from utils.parameters import Params
from utils.data_loader import CustomDataset
from utils.set_seed import SetSeed
from utils.inspector import ModelInspector


class Inference(object):
    def __init__(self, model, test_loader):
        super().__init__()
        self.set_seed = SetSeed(42)
        self.set_seed.set_seed()
        self.device = Params.device
        self.model = model
        self.test_loader = test_loader
        self.save_dir = Params.save_dir
        self.best_model_path = Params.best_model_path

    def measure_snr_db(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        *,
        time_axis: int = -1,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Measure per-channel SNR over time only. Returns shape (B,1) or (B,C,1)."""
        if time_axis != -1:
            perm = list(range(x_clean.ndim))
            perm[time_axis], perm[-1] = perm[-1], perm[time_axis]
            x_clean = x_clean.permute(*perm)
            x_noisy = x_noisy.permute(*perm)

        dims = (-1,)
        p_sig = x_clean.pow(2).mean(dim=dims, keepdim=True)
        p_noise = (x_noisy - x_clean).pow(2).mean(dim=dims, keepdim=True)
        snr = 10.0 * torch.log10(
            torch.clamp(p_sig, min=eps) / torch.clamp(p_noise, min=eps)
        )
        return snr

    def add_awgn_correlated(self, x, snr_db, time_axis=-1, eps=1e-12, seed=42):
        # shape (B,C,T) or (B,T)
        if time_axis != -1:
            perm = list(range(x.ndim))
            perm[time_axis], perm[-1] = perm[-1], perm[time_axis]
            x = x.permute(*perm)
        dims = (-1,)
        p_sig = x.pow(2).mean(dim=dims, keepdim=True)  # (B, C?, 1)
        snr_lin = 10.0 ** (snr_db / 10.0)
        p_noise = torch.clamp(p_sig / snr_lin, min=eps)

        # one noise trace per (B,T), then broadcast to channels → fully correlated across C
        if x.ndim == 3:
            B, C, T = x.shape
            gen = torch.Generator(device=x.device)
            if seed is not None:
                gen.manual_seed(seed)
            base = torch.randn(B, 1, T, device=x.device, dtype=x.dtype, generator=gen)
            noise = base.expand(B, C, T).contiguous()
        else:
            noise = torch.randn_like(x)

        p_now = noise.pow(2).mean(dim=dims, keepdim=True)
        noise = noise * torch.sqrt(p_noise / (p_now + eps))
        y = x + noise

        if time_axis != -1:
            inv = list(range(y.ndim))
            inv[time_axis], inv[-1] = inv[-1], inv[time_axis]
            y = y.permute(*inv)
        return y

    def sort_test_loader(self, test_loader):
        # Extract data from DataLoader
        data_list = []
        for data, target in test_loader:
            data_list.append({"waveform": data, "target": target})

        # Sort data based on targets
        sorted_data_list = sorted(data_list, key=lambda x: x["target"][0].item())

        # Create a new CustomDataset with the sorted data
        sorted_dataset = CustomDataset(sorted_data_list)

        # Create a new DataLoader with the sorted dataset
        sorted_test_loader = DataLoader(
            sorted_dataset, batch_size=test_loader.batch_size, shuffle=False
        )

        return sorted_test_loader

    def inference(self) -> None:
        """
        Perform inference on the test dataset using the trained model
        """

        if Params.finetune_mode and Params.num_epochs == 0:
            logging.info("Testing the model with zero-shot learning...")
            best_model_path = Params.pretrained_model_path
        else:
            best_model_path = os.path.join(self.save_dir, self.best_model_path)

        self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        all_predictions = []
        all_targets = []

        # if not Params.cross_validation_mode:
        #     logging.info("Sort the test dataset...")
        #     # Sort the test_loader
        #     self.test_loader = self.sort_test_loader(self.test_loader)

        inspector = ModelInspector(self.model)
        # inspector.register_hooks()
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.view(-1, 1)

                # Use this add_noise = True for quick test with noise for SNR measurement at inference,
                # will add as Params later
                add_noise = False
                if add_noise:
                    SNR_lvl = 10 # SNR level in dB
                    logging.info(f"SNR is: {SNR_lvl}")
                    noisy = self.add_awgn_correlated(inputs, SNR_lvl, time_axis=-1)
                    snr_meas = self.measure_snr_db(inputs, noisy, time_axis=-1)
                    logging.info(f"SNR_MEASUREMENT: {snr_meas}")
                    output = self.model(noisy)
                else:
                    output = self.model(inputs)

                predicted_labels = output.cpu().numpy()
                batch_targets = labels.cpu().numpy()
                all_predictions.extend(predicted_labels)
                all_targets.extend(batch_targets)

        # inspector.save_weights()

        # for i, block in enumerate(self.model.conformer1.layers):
        #     if hasattr(block, "attn_weights") and block.attn_weights is not None:
        #         logging.info("Attention")
        #         inspector.visualize_attention(block.attn_weights[0], f"conformer_block_{i}")

        # # Clean up
        # inspector.remove_hooks()

        # Convert lists to numpy arrays for easier manipulation
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        if not Params.cross_validation_mode:
            sorted_dict = sorted(zip(all_predictions, all_targets), key=lambda x: x[1])
            all_predictions, all_targets = zip(*sorted_dict)

        if Params.print_first_10th_predictions:
            logging.info(all_predictions[:10])
            logging.info(all_targets[:10])

        def calculate_metrics(predictions, targets):
            S = len(targets)
            # Ensure the predictions and targets lists have the same length
            if S != len(predictions):
                raise ValueError("Predictions and targets must have the same length")

            total_absolute_error = 0
            total_squared_error = 0

            for yi, fxi in zip(targets, predictions):
                absolute_error = abs(yi - fxi)
                squared_error = (yi - fxi) ** 2
                # Accumulate the absolute error for MAE calculation
                total_absolute_error += absolute_error
                # Accumulate the squared error for MSE calculation
                total_squared_error += squared_error

            mae = total_absolute_error / S
            mse = total_squared_error / S

            return mae, mse

        mae, mse = calculate_metrics(all_predictions, all_targets)
        logging.info(f"MAE: {mae}")
        logging.info(f"MSE: {mse}")

        # IF S59 and test all 6 folds, rearrange for a nicer plot (stack 6 folds instead of plot 1 by 1):
        if "s59" in Params.dataset_path and Params.cross_validation_mode == True:
            # folds = list of six 1-D arrays, all length L
            # e.g., folds = [fold1, fold2, fold3, fold4, fold5, fold6]
            n_folds = Params.total_folds
            fold_len = all_predictions.shape[0] // n_folds

            # Reshape to (folds, fold_len)
            preds = all_predictions.reshape(n_folds, fold_len)
            targs = all_targets.reshape(n_folds, fold_len)

            # Transpose to (fold_len, folds) and flatten row-wise
            all_predictions = preds.T.reshape(-1)
            all_targets = targs.T.reshape(-1)

        # Select every 10th point
        indices = np.arange(1, len(all_targets) + 1)
        indices_10th = indices[::10]
        all_targets_10th = all_targets[::10]
        all_predictions_10th = all_predictions[::10]

        # Ensure the arrays are 1-dimensional
        all_targets_10th = np.squeeze(all_targets_10th)
        all_predictions_10th = np.squeeze(all_predictions_10th)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the true labels in blue
        ax.scatter(
            indices_10th, all_targets_10th, c="b", label="Derived Ground Truth", s=8
        )
        ax.scatter(
            indices_10th,
            all_predictions_10th,
            c="r",
            marker="x",
            label="Predicted Labels",
            s=8,
        )

        # Add shaded area for ±5% range
        lower_bound = all_targets_10th * 0.95
        upper_bound = all_targets_10th * 1.05
        ax.fill_between(
            indices_10th,
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.2,
            label="±5% Range",
        )

        # Add labels and legend
        ax.set_xlabel("Index")
        ax.set_ylabel("Range (Km)")
        ax.set_title("Ground Truth vs. Predicted Labels")
        ax.legend()

        plt.savefig(f"{Params.prediction_output_image_plot}")
