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

                # # some changes:
                # if inputs.shape[2] == 14784:
                #     pass
                # elif inputs.shape[2] == 16192:
                #     inputs = nn.Conv1d(16192, 14784, 1)(inputs)

                # _, _, output = self.model(inputs)
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

        # def calculate_pcl5_mae(predictions, targets):
        #     S = len(targets)

        #     # Ensure the predictions and targets lists have the same length
        #     if S != len(predictions):
        #         raise ValueError("Predictions and targets must have the same length")

        #     # Initialize counters for PCL-5% and MAE
        #     pcl5_count = 0
        #     total_absolute_error = 0

        #     for yi, fxi in zip(targets, predictions):
        #         absolute_error = abs(yi - fxi)
        #         percentage_error = (absolute_error / yi) * 100

        #         # Check if the percentage error is within 5%
        #         if percentage_error <= 5:
        #             pcl5_count += 1

        #         # Accumulate the absolute error for MAE calculation
        #         total_absolute_error += absolute_error

        #     # Calculate PCL-5%
        #     pcl5 = (pcl5_count / S) * 100

        #     # Calculate MAE
        #     mae = total_absolute_error / S

        #     return pcl5, mae

        # pcl5, mae = calculate_pcl5_mae(all_predictions, all_targets)

        def calculate_metrics(predictions, targets):
            S = len(targets)

            # Ensure the predictions and targets lists have the same length
            if S != len(predictions):
                raise ValueError("Predictions and targets must have the same length")

            # Initialize counters for PCL-5%, PCL-10%, MAE, and MSE
            pcl5_count = 0
            pcl10_count = 0
            total_absolute_error = 0
            total_squared_error = 0

            for yi, fxi in zip(targets, predictions):
                absolute_error = abs(yi - fxi)
                squared_error = (yi - fxi) ** 2
                percentage_error = (absolute_error / yi) * 100

                # Check if the percentage error is within 5%
                if percentage_error <= 5:
                    pcl5_count += 1

                # Check if the percentage error is within 10%
                if percentage_error <= 10:
                    pcl10_count += 1

                # Accumulate the absolute error for MAE calculation
                total_absolute_error += absolute_error

                # Accumulate the squared error for MSE calculation
                total_squared_error += squared_error

            # Calculate PCL-5%
            pcl5 = (pcl5_count / S) * 100

            # Calculate PCL-10%
            pcl10 = (pcl10_count / S) * 100

            # Calculate MAE
            mae = total_absolute_error / S

            # Calculate MSE
            mse = total_squared_error / S

            return pcl5, pcl10, mae, mse

        pcl5, pcl10, mae, mse = calculate_metrics(all_predictions, all_targets)
        logging.info(f"PCL-5%: {pcl5}%")
        logging.info(f"PCL-10%: {pcl10}%")
        logging.info(f"MAE: {mae}")
        logging.info(f"MSE: {mse}")

        # indices = np.arange(1, len(all_targets) + 1)

        # fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the true labels in blue
        # ax.scatter(indices, all_targets, c="b", label="Derived Ground Truth", s=8)
        # ax.scatter(
        #     indices, all_predictions, c="r", marker="x", label="Predicted Labels", s=8
        # )

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
