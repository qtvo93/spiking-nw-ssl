# Description: Loss class for calculating the loss of the model
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import snntorch.functional as SF

from utils.parameters import Params
from utils.set_seed import SetSeed


class Loss(object):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.device = Params.device
        self.set_seed = SetSeed(42)
        self.set_seed.set_seed()

    def calculate_mse_loss(self, output, target) -> torch.Tensor:
        # wave_speed, frequency, alpha, beta, gamma = 2.51, 1500, 1.0, 1.0, 1.0
        # mse = self.mse_loss(output, target)
        # # helmholtz = self.helmholtz_loss(output, inputs, wave_speed, frequency)
        # energy_conservation = self.energy_conservation_loss(output, target)
        # # localization = self.source_localization_loss(output, target, propagation_model)

        # total_loss = mse + energy_conservation
        # return total_loss
        # loss_fnc = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        return self.mse_loss(output, target)
        # return loss_fnc(output, target)

    def helmholtz_loss(self, pred, inputs, wave_speed, frequency):
        """
        Enforce Helmholtz equation: (∇² + k²) u = 0, where k = 2πf / c
        """
        laplacian = torch.autograd.grad(
            torch.autograd.grad(
                pred, inputs, grad_outputs=torch.ones_like(pred), create_graph=True
            )[0],
            inputs,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
        )[0]
        k = 2 * torch.pi * frequency / wave_speed
        return torch.mean((laplacian + k**2 * pred) ** 2)

    def energy_conservation_loss(self, pred, target):
        """
        Penalize energy conservation violations: energy in should equal energy out
        """
        return torch.mean((torch.sum(pred, dim=-1) - torch.sum(target, dim=-1)) ** 2)

    def source_localization_loss(self, pred, true_range, propagation_model):
        """
        Penalize deviations from known propagation models
        """
        physics_based_output = propagation_model(true_range)
        return torch.mean((pred - physics_based_output) ** 2)

    def calculate_mse_loss_2(self, pred, target, eps=1e-8):
        """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss."""
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, but got {type(pred)} and {type(target)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape}, target {target.shape}"
            )

        target_energy = torch.sum(target**2, dim=-1, keepdim=True) + eps
        scale = torch.sum(pred * target, dim=-1, keepdim=True) / target_energy
        target_scaled = scale * target
        noise = pred - target_scaled
        si_sdr = 10 * torch.log10(
            torch.sum(target_scaled**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
        )

        return si_sdr.mean()

    def validation_loss(self, model, val_loader) -> float:
        """
        Calculate the validation loss of the model

        Args:
            model: model to validate
            val_loader: validation data loader

        Returns:
            val_loss: validation loss
        """
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.view(-1, 1)
                #   # some changes:
                # if inputs.shape[2] == 14784:
                #     pass
                # elif inputs.shape[2] == 16192:
                #     inputs = nn.Conv1d(16192, 14784, 1)(inputs)
                # _, _, outputs = model(inputs)
                outputs = model(inputs)
                loss = self.calculate_mse_loss(outputs, labels.float())
                running_loss += loss.item()
            val_loss = running_loss / len(val_loader)

        return val_loss

    # Speed of sound model (can be depth-dependent)
    def sound_speed(self, depth):
        # Simple linear speed model: Sound speed increases with depth
        # You can replace this with a more realistic profile.
        return 1500 + 0.5 * depth  # Speed in m/s (just an example)

    # Ray tracing propagation model
    def trace_ray(self, source, receiver, depth_profile, num_rays=100):
        rays = []
        ray_positions = np.linspace(source[1], receiver[1], num_rays)

        for ray_pos in ray_positions:
            # Simple straight line approximation (for simplicity, ignoring refraction)
            path = [source]
            for depth in depth_profile:
                # Simulate a "reflection" at each depth layer
                # (In reality, this would be a more complex calculation involving Snell's law)
                sound_speed_at_depth = self.sound_speed(depth)
                new_pos = [ray_pos, depth]
                path.append(new_pos)
            rays.append(np.array(path))
        return rays

    # Visualization function
    def plot_rays(self, rays, source, receiver):
        plt.figure(figsize=(10, 6))
        for ray in rays:
            plt.plot(ray[:, 1], ray[:, 0], label="Ray Path")

        plt.scatter([source[1]], [source[0]], color="red", label="Source", zorder=5)
        plt.scatter(
            [receiver[1]], [receiver[0]], color="blue", label="Receiver", zorder=5
        )
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.title("Ray Tracing Model for Acoustic Propagation")
        plt.gca().invert_yaxis()  # Depth increases downward
        plt.legend()
        plt.show()
