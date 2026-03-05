# Description: Trainer class for setting up and training the model
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-05
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

import torch
import logging

# from snntorch import utils

from utils.set_seed import SetSeed


class Trainer(object):
    def __init__(
        self,
        best_model_path,
        optimizer,
        criterion,
        num_epochs,
        device,
        model,
        train_loader,
        val_loader,
    ):
        super().__init__()
        self.set_seed = SetSeed(42)
        self.set_seed.set_seed()

        self.best_model_path = best_model_path
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_supervised_model(self) -> None:
        """
        Train the model using the training dataset and validate using the validation dataset

        Args:
            model: model to train
            train_loader: training data loader
            val_loader: validation data loader
            criterion: loss function
            optimizer: optimizer
            num_epochs: number of epochs

        Returns:
            None
        """
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            self.model.train()
            # utils.reset(self.model)
            total_loss = 0
            for batch, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.view(-1, 1)
                # print(inputs.shape)
                # # some changes: torch.Size([32, 75, 16192])
                # if inputs.shape[2] == 14784:
                #     pass
                # elif inputs.shape[2] == 16192:
                #     inputs = nn.Conv1d(16192, 14784, 1)(inputs)

                output = self.model(inputs)

                loss = self.criterion.calculate_mse_loss(output, labels.float())
                if torch.isnan(loss):
                    logging.info(f"NaN detected in loss at epoch {epoch}")
                    break

                regularization_type = None
                lambda_reg = 0.01
                # Apply L1 regularization
                if regularization_type == "L1":
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss += lambda_reg * l1_norm

                # Apply L2 regularization
                elif regularization_type == "L2":
                    l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                    loss += lambda_reg * l2_norm

                loss.backward()
                self.optimizer.step()

                # Check for NaN in model parameters and gradients
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        logging.info(f"NaN detected in parameters: {name}")
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logging.info(f"NaN detected in gradients: {name}")

                total_loss += loss.item()

            val_loss = self.criterion.validation_loss(self.model, self.val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info(
                    f"Saved the best model with validation loss: {best_val_loss:.4f}"
                )
            logging.info(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {total_loss / len(self.train_loader)}, Validation Loss: {val_loss}"
            )
            # scheduler.step(val_loss)

    # def train_supervised_model(self):
    #     best_val_loss = float("inf")
    #     domain_criterion = nn.BCELoss()

    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         for batch, data in enumerate(self.train_loader):

    #             inputs, labels = data
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)
    #             labels = labels.view(-1, 1)

    #             output, domain_outputs = self.model(inputs)

    #             reg_loss = self.criterion.calculate_mse_loss(output, labels.float())
    #             if torch.isnan(reg_loss):
    #                 logging.info(f"NaN detected in loss at epoch {epoch}")
    #                 break

    #             self.optimizer.zero_grad()
    #             # Domain loss (force domain confusion)
    #             domain_labels = torch.zeros_like(domain_outputs).to(self.device)  # Simulated = 0
    #             domain_loss = domain_criterion(domain_outputs, domain_labels)

    #             # Total loss
    #             loss = reg_loss + 0.1 * domain_loss  # Small weight for domain loss
    #             loss.backward()
    #             self.optimizer.step()

    #             # Check for NaN in model parameters and gradients
    #             for name, param in self.model.named_parameters():
    #                 if torch.isnan(param).any():
    #                     logging.info(f"NaN detected in parameters: {name}")
    #                 if param.grad is not None and torch.isnan(param.grad).any():
    #                     logging.info(f"NaN detected in gradients: {name}")

    #             total_loss += loss.item()

    #         val_loss = self.criterion.validation_loss(self.model, self.val_loader)
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             torch.save(self.model.state_dict(), self.best_model_path)
    #             logging.info(
    #                 f"Saved the best model with validation loss: {best_val_loss:.4f}"
    #             )
    #         logging.info(
    #             f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {total_loss / len(self.train_loader)}, Validation Loss: {val_loss}"
    #         )

    # def train_supervised_model(self):
    #     best_val_loss = float("inf")
    #     lambda_mmd = 0.5
    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         for source_data, target_data in zip(self.train_loader, self.val_loader):
    #             source_inputs, source_labels = source_data
    #             target_inputs, _ = target_data  # Unlabeled real data

    #             source_inputs, source_labels = source_inputs.to(self.device), source_labels.view(-1, 1).to(self.device)
    #             target_inputs = target_inputs.to(self.device)

    #             self.optimizer.zero_grad()
    #             # Forward pass on source and target
    #             x_mel_s, x_gcc_s, _ = self.model(source_inputs)
    #             x_mel_t, x_gcc_t, _ = self.model(target_inputs)

    #             # print(x_mel_s.shape, x_mel_t.shape)
    #             # print(x_mel_s)
    #             # Compute standard regression loss on source
    #             _, _, preds = self.model(source_inputs)
    #             regression_loss = self.criterion.calculate_mse_loss(preds, source_labels.float())

    #             # Compute MMD loss between source and target
    #             mmd_loss_mel = self.compute_mmd(x_mel_s, x_mel_t)
    #             mmd_loss_gcc = self.compute_mmd(x_gcc_s, x_gcc_t)
    #             mmd_loss = (mmd_loss_mel + mmd_loss_gcc) / 2  # Average MMD loss

    #             # Total loss
    #             total_loss = regression_loss + lambda_mmd * mmd_loss
    #             total_loss.backward()
    #             self.optimizer.step()

    #             # Check for NaN in model parameters and gradients
    #             for name, param in self.model.named_parameters():
    #                 if torch.isnan(param).any():
    #                     logging.info(f"NaN detected in parameters: {name}")
    #                 if param.grad is not None and torch.isnan(param.grad).any():
    #                     logging.info(f"NaN detected in gradients: {name}")

    #             total_loss += total_loss.item()

    #         val_loss = self.criterion.validation_loss(self.model, self.val_loader)
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             torch.save(self.model.state_dict(), self.best_model_path)
    #             logging.info(
    #                 f"Saved the best model with validation loss: {best_val_loss:.4f}"
    #             )
    #         logging.info(
    #             f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {total_loss / len(self.train_loader)}, Validation Loss: {val_loss}"
    #         )

    # def compute_mmd(self, x_source, x_target, sigma=1.0):
    #     """
    #     Computes the Maximum Mean Discrepancy (MMD) using an RBF kernel.
    #     :param x_source: Source domain features (batch_size, feature_dim)
    #     :param x_target: Target domain features (batch_size, feature_dim)
    #     :param sigma: Bandwidth for the RBF kernel
    #     :return: MMD loss
    #     """
    #     x_source = x_source.view(x_source.size(0), -1)  # Flatten if needed
    #     x_target = x_target.view(x_target.size(0), -1)  # Flatten if needed

    #     def rbf_kernel(x, y, sigma):
    #         xx = torch.matmul(x, x.T)
    #         yy = torch.matmul(y, y.T)
    #         xy = torch.matmul(x, y.T)

    #         x_norm = torch.diagonal(xx).unsqueeze(1)
    #         y_norm = torch.diagonal(yy).unsqueeze(1)

    #         k_xx = torch.exp(-((x_norm + x_norm.T - 2 * xx) / (2 * sigma ** 2)))
    #         k_yy = torch.exp(-((y_norm + y_norm.T - 2 * yy) / (2 * sigma ** 2)))
    #         k_xy = torch.exp(-((x_norm + y_norm.T - 2 * xy) / (2 * sigma ** 2)))

    #         return k_xx, k_yy, k_xy

    #     k_xx, k_yy, k_xy = rbf_kernel(x_source, x_target, sigma)

    #     mmd_loss = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    #     return mmd_loss
