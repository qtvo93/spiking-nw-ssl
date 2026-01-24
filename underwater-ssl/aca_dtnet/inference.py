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
from aca_dtnet.kalman import KalmanCV_CVA


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


    def add_awgn(
        self,
        x: torch.Tensor,
        snr_db: float,
        *,
        time_axis: int = -1,
        eps: float = 1e-12,
        seed: Optional[int] = 42,
        clamp_silent: float = 0.0,  # e.g., 1e-6 to avoid blowing up silence
    ) -> torch.Tensor:
        """
        Add white Gaussian noise at target SNR_dB.
        - Per-channel SNR if x has a channel dim (keeps channel dim, averages over time only).
        - For 2D (B, T) input, averages over time only.

        Args:
            x: (B, T) or (B, C, T) or similar; time axis specified by `time_axis`.
            snr_db: target SNR in dB (power SNR).
            time_axis: which axis is time; default last (-1).
            clamp_silent: floor for signal RMS; set >0 to avoid huge noise on near-silent segments.

        Returns:
            x + noise with the requested SNR.
        """
        assert x.is_floating_point(), "x must be float"
        # Move time axis to the end for convenience
        if time_axis != -1:
            perm = list(range(x.ndim))
            perm[time_axis], perm[-1] = perm[-1], perm[time_axis]
            x = x.permute(*perm)

        # After permute, shapes:
        # (B, T) -> (B, T)
        # (B, C, T) -> (B, C, T)
        # Compute power over time only (keep dims to broadcast)
        dims = (-1,)
        p_sig = x.pow(2).mean(dim=dims, keepdim=True)  # (B,1) or (B,C,1)

        if clamp_silent > 0.0:
            p_sig = torch.clamp(p_sig, min=clamp_silent**2)

        snr_lin = 10.0 ** (snr_db / 10.0)
        p_noise = torch.clamp(p_sig / snr_lin, min=eps)  # (B,1) or (B,C,1)

        # Make white Gaussian noise on the same device/dtype
        if seed is not None:
            try:
                gen = torch.Generator(device=x.device)
            except TypeError:
                gen = torch.Generator()
            gen.manual_seed(seed)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        else:
            noise = torch.randn_like(x)

        # Normalize noise power per (B, C) or per (B) before scaling
        p_noise_now = noise.pow(2).mean(dim=dims, keepdim=True)  # (B,1) or (B,C,1)
        noise = noise * torch.sqrt(p_noise / (p_noise_now + eps))

        y = x + noise

        # Put axes back if we permuted
        if time_axis != -1:
            inv = list(range(y.ndim))
            inv[time_axis], inv[-1] = inv[-1], inv[time_axis]
            y = y.permute(*inv)

        return y


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
        snr = 10.0 * torch.log10(torch.clamp(p_sig, min=eps) / torch.clamp(p_noise, min=eps))
        return snr

    def add_awgn_correlated(self, x, snr_db, time_axis=-1, eps=1e-12, seed=42):
        # shape (B,C,T) or (B,T)
        if time_axis != -1:
            perm = list(range(x.ndim)); perm[time_axis], perm[-1] = perm[-1], perm[time_axis]
            x = x.permute(*perm)
        dims = (-1,)
        p_sig = x.pow(2).mean(dim=dims, keepdim=True)              # (B, C?, 1)
        snr_lin = 10.0 ** (snr_db / 10.0)
        p_noise = torch.clamp(p_sig / snr_lin, min=eps)

        # one noise trace per (B,T), then broadcast to channels → fully correlated across C
        if x.ndim == 3:
            B, C, T = x.shape
            gen = torch.Generator(device=x.device); 
            if seed is not None: gen.manual_seed(seed)
            base = torch.randn(B, 1, T, device=x.device, dtype=x.dtype, generator=gen)
            noise = base.expand(B, C, T).contiguous()
        else:
            noise = torch.randn_like(x)

        p_now = noise.pow(2).mean(dim=dims, keepdim=True)
        noise = noise * torch.sqrt(p_noise / (p_now + eps))
        y = x + noise

        if time_axis != -1:
            inv = list(range(y.ndim)); inv[time_axis], inv[-1] = inv[-1], inv[time_axis]
            y = y.permute(*inv)
        return y
    
    def add_awgn_partially_correlated(
        self,
        x: torch.Tensor, snr_db: float, rho: float = 0.8, *,
        time_axis: int = -1, eps: float = 1e-12, seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """Add white Gaussian noise with per-channel SNR and target inter-channel correlation rho."""
        assert 0.0 <= rho <= 1.0 and x.is_floating_point()
        # Put time last
        if time_axis != -1:
            perm = list(range(x.ndim)); perm[time_axis], perm[-1] = perm[-1], perm[time_axis]
            x = x.permute(*perm)
        # Shapes: (B,T) or (B,C,T)
        dims = (-1,)
        p_sig = x.pow(2).mean(dim=dims, keepdim=True)                            # (B,1,1) or (B,C,1)
        snr_lin = 10.0 ** (snr_db / 10.0)
        p_noise = torch.clamp(p_sig / snr_lin, min=eps)                          # desired noise power per ch

        BCT = x.shape
        device, dtype = x.device, x.dtype
        gen = None
        if seed is not None:
            try:
                gen = torch.Generator(device=device)
            except TypeError:
                gen = torch.Generator()
            gen.manual_seed(seed)

        if x.ndim == 3:
            B, C, T = BCT
            # shared component: one waveform per batch, broadcast to all channels
            z_shared = torch.randn(B, 1, T, device=device, dtype=dtype, generator=gen)
            # independent component: one per channel
            z_indep  = torch.randn(B, C, T, device=device, dtype=dtype, generator=gen)
            # target per-channel std for noise
            sigma = torch.sqrt(p_noise)                                          # (B,C,1)
            # mix shared + independent; this construction gives Corr_ij = rho
            noise = ( (rho**0.5) * sigma * z_shared + (1 - rho)**0.5 * sigma * z_indep )
        else:
            # (B,T): correlation concept doesn’t apply; just AWGN
            z = torch.randn_like(x) if gen is None else torch.randn(x.shape, device=device, dtype=dtype, generator=gen)
            sigma = torch.sqrt(p_noise)                                          # (B,1)
            noise = sigma * z

        y = x + noise
        # Restore original axis order
        if time_axis != -1:
            inv = list(range(y.ndim)); inv[time_axis], inv[-1] = inv[-1], inv[time_axis]
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

        dt = 1
        sigma_a = 0.2
        r_std   = 80.0          # ≈ dev RMSE (meters); or tune
        kf = KalmanCV_CVA(dt=dt, sigma_a=sigma_a, r_std=r_std,
        device=self.device, dtype=torch.float32)

        import cvxpy as cp
        import scipy.sparse as sp

        def _solve_cvx(prob, verbose=False):
            # Try OSQP first
            try:
                prob.solve(
                    solver=cp.OSQP,
                    polish=True,
                    eps_abs=1e-7,
                    eps_rel=1e-7,
                    max_iter=200000,
                    adaptive_rho=True,
                    verbose=verbose
                )
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return
            except Exception:
                pass
            # ECOS fallback (interior-point)
            try:
                prob.solve(
                    solver=cp.ECOS,
                    abstol=1e-9, reltol=1e-9, feastol=1e-9,
                    max_iters=20000, verbose=verbose
                )
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return
            except Exception:
                pass
            # SCS fallback (first-order; robust on tough cases)
            prob.solve(
                solver=cp.SCS,
                eps=1e-5, max_iters=100000,
                acceleration_lookback=20,
                scale=1.0, verbose=verbose
            )

        def smooth_range_physics_km_stable(
            z_km,                 # shape (T,) or (T,1)
            v_max_km_s=0.0025,    # 2.5 m/s
            a_max_km_s2=0.0005,   # 0.5 m/s^2
            r_min_km=0.0,
            r_max_km=20.0,
            doppler_sign=None,    # optional array length T-1: +1 recede, -1 approach, 0 free
            trust=1.0,
            iter_robust=2,
            dt_s=1.0,
            lambda_v=1e3,         # soft penalty for speed slack
            lambda_a=1e3,         # soft penalty for accel slack
            ridge=1e-9,           # tiny L2 on r for conditioning
            verbose=False
        ):
            # ---- flatten & cast ----
            z_km = np.asarray(z_km, dtype=np.float64).reshape(-1)
            T = z_km.shape[0]
            if T < 3:
                return z_km.copy()

            # ---- scale to meters for numerics ----
            KM = 1000.0
            z_m = z_km * KM
            r_min_m, r_max_m = r_min_km*KM, r_max_km*KM
            v_bound_m = v_max_km_s * dt_s * KM
            a_bound_m = a_max_km_s2 * (dt_s**2) * KM

            # ---- sparse banded difference operators ----
            # D1: (T-1) x T first difference
            I_T = sp.eye(T, format="csc")
            D1 = I_T - sp.eye(T, k=1, format="csc")
            D1 = D1[:T-1, :]
            # D2: (T-2) x T second difference
            I_T1 = sp.eye(T-1, format="csc")
            D1m = I_T1 - sp.eye(T-1, k=1, format="csc")
            D1m = D1m[:T-2, :]
            D2 = D1m @ D1

            # ---- robust weights init ----
            w = np.ones(T, dtype=np.float64)

            def hampel_weights(res, k=3.0):
                med = np.median(res)
                mad = np.median(np.abs(res - med)) + 1e-12
                z = np.abs(res - med) / (1.4826 * mad)
                w = np.ones_like(z)
                m = z > k
                w[m] = k / z[m]
                return w

            # ---- optional monotonicity mask from doppler_sign (length T-1) ----
            use_mono = (doppler_sign is not None) and (len(doppler_sign) == T-1)
            doppler_sign = np.asarray(doppler_sign, dtype=np.int8) if use_mono else None

            r_m = z_m.copy()

            for _ in range(iter_robust):
                r_var = cp.Variable(T)

                # Weighted data fidelity + tiny ridge
                # (Use diagonal via elementwise multiply to keep it sparse)
                data_term = cp.sum_squares(cp.multiply(w * trust, r_var - z_m)) + ridge * cp.sum_squares(r_var)

                # Smoothness penalty (accel L2)
                smooth_term = cp.sum_squares(D2 @ r_var)

                # Soft constraint slacks
                s_v = cp.Variable(T-1, nonneg=True)
                s_a = cp.Variable(T-2, nonneg=True)

                constraints = [
                    r_var >= r_min_m,
                    r_var <= r_max_m,
                    # speed (soft): |D1 r| <= v_bound + s_v
                    D1 @ r_var <=  v_bound_m + s_v,
                    D1 @ r_var >= -v_bound_m - s_v,
                    # accel (soft): |D2 r| <= a_bound + s_a
                    D2 @ r_var <=  a_bound_m + s_a,
                    D2 @ r_var >= -a_bound_m - s_a,
                ]

                # Optional monotonicity: enforce sign on D1 r
                if use_mono:
                    # sign: +1 => Δr >= 0 ; -1 => Δr <= 0 ; 0 => no constraint
                    idx_pos = np.where(doppler_sign > 0)[0]
                    idx_neg = np.where(doppler_sign < 0)[0]
                    if idx_pos.size:
                        constraints += [(D1[idx_pos, :] @ r_var) >= 0]
                    if idx_neg.size:
                        constraints += [(D1[idx_neg, :] @ r_var) <= 0]

                obj = cp.Minimize(data_term + smooth_term + lambda_v*cp.sum_squares(s_v) + lambda_a*cp.sum_squares(s_a))
                prob = cp.Problem(obj, constraints)
                _solve_cvx(prob, verbose=verbose)

                r_m = np.asarray(r_var.value, dtype=np.float64)
                # IRLS reweighting
                w = hampel_weights(z_m - r_m, k=3.0)

            # back to km
            return r_m / KM

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
                add_noise = True
                if add_noise:
                    SNR_lvl = -15
                    logging.info(f"SNR is: {SNR_lvl}")
                    p_value = 1
                    # noisy = self.add_awgn(inputs, SNR_lvl, time_axis=-1, clamp_silent=0.0)
                    noisy = self.add_awgn_correlated(inputs, SNR_lvl, time_axis=-1)
                    # noisy = self.add_awgn_partially_correlated(inputs, snr_db=SNR_lvl, rho=p_value, time_axis=-1, seed=42)

                    snr_meas = self.measure_snr_db(inputs, noisy, time_axis=-1)
                    # logging.info(f"SNR_MEASUREMENT: {snr_meas}")
                    # _, _, output = self.model(inputs)
                    output = self.model(noisy)
                else:
                    output = self.model(inputs)

                # reduce_dims = tuple(range(1, snr_meas.ndim))
                # snr_meas_db = snr_meas.mean(dim=reduce_dims)            # (B,)

                # mu_m = output * 1000.0
                # snr_ref  = 12        
                # base_std = 1            # meters;
                # scale = torch.pow(10.0, -(snr_meas_db - snr_ref) / 20.0)  # (B,)
                # R_t_std_m = torch.clamp(base_std * scale, 0.5, 50.0)              # (B,) std in meters
                # kf = KalmanCV_CVA(dt=1.0, sigma_a=0.2, r_std=base_std,
                # device=mu_m.device, dtype=mu_m.dtype)
                # r_filt_m = kf.run(mu_m, R_t=None, smooth=False)   # (B,)
                # r_filt_km = r_filt_m / 1000.0
                # r_filt_batch = 
                
                # r_filt_batch = kf.run(output, R_t=None, smooth=False)
                # all_predictions.extend(r_filt_batch.detach().cpu().numpy())   # <— KF 
                
                # ---- smoother usage ----
                z_pred_km = np.asarray(output.detach().cpu().numpy())
                pilot_freqs_hz = np.array([
                    49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85,
                    88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124,
                    127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163,
                    166, 169, 172, 175, 178, 198, 201, 204, 207, 210, 213, 232, 235,
                    238, 241, 244, 247, 280, 283, 286, 289, 292, 295, 335, 338, 341,
                    344, 347, 350, 385, 388, 391, 394, 397, 400
                ], dtype=float)  
                doppler_hz = np.asarray(pilot_freqs_hz * (2.5/1500.0))

                def doppler_sign_from_doppler_hz(doppler_hz_t, smooth_win=5, pos_thresh=0.01, neg_thresh=-0.01):
                    """
                    doppler_hz_t: length T time series of instantaneous Doppler shift (Hz)
                    Returns doppler_sign of length T-1 in {+1, -1, 0}.
                    """
                    x = np.asarray(doppler_hz_t, float).reshape(-1)
                    # Smooth tiny jitters
                    if smooth_win > 1:
                        kern = np.ones(smooth_win) / smooth_win
                        x = np.convolve(x, kern, mode='same')
                    # Sign by thresholds
                    s = np.zeros_like(x, dtype=int)
                    s[x > pos_thresh] = +1
                    s[x < neg_thresh] = -1
                    # Map to T-1 (use value at left endpoint of each interval)
                    return s[:-1]
                
                doppler_sign = doppler_sign_from_doppler_hz(doppler_hz)
                r_smooth_km = smooth_range_physics_km_stable(
                    z_pred_km,
                    v_max_km_s=0.003,   # 2.5 m/s
                    a_max_km_s2=0.001,  # 0.5 m/s^2
                    r_min_km=0.903,
                    r_max_km=8.648,       
                    doppler_sign=None,
                    trust=1.0,
                    iter_robust=2,
                    dt_s=1.0,
                    lambda_v=1e2,         # soft penalty for speed slack
                    lambda_a=1e2,         # soft penalty for accel slack
                    ridge=1e-9,           # tiny L2 on r for conditioning
                    verbose=False
                )
                # all_predictions.extend(r_filt_km.detach().cpu().numpy())
                all_predictions.extend(r_smooth_km)
                # all_predictions.extend(output.detach().cpu().numpy())
                all_targets.extend(labels.detach().cpu().numpy())

                # predicted_labels = output.cpu().numpy()
                # batch_targets = labels.cpu().numpy()
                # all_predictions.extend(predicted_labels)
                # all_targets.extend(batch_targets)


                # # all_predictions, all_targets are in KM in
                # pred_km = np.asarray(all_predictions).reshape(-1)
                # true_km = np.asarray(all_targets).reshape(-1)

                # # RMSE in meters
                # err_m   = (pred_km - true_km) * 1000.0
                # rmse_m  = float(np.sqrt(np.mean(err_m**2)))
                # mae_m   = float(np.mean(np.abs(err_m)))
                # print(f"[DEV] RMSE = {rmse_m:.2f} m, MAE = {mae_m:.2f} m")
        
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
