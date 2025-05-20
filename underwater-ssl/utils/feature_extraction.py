# Description: This file contains the FeatureExtraction class which is used to extract features from the audio data.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import numpy as np
import librosa
import pandas as pd
import torch
import librosa.display
import math
import pickle
import logging
import joblib
import torchaudio.transforms as T
from scipy.interpolate import interp1d
from utils.parameters import Params


class FeatureExtraction(object):
    def __init__(
        self,
        data_augmentation: str,
    ) -> None:
        super().__init__()
        self.data_augmentation = data_augmentation
        self.csv_file_path = Params.csv_file_path
        self.sproul_text_file_path = Params.sproul_text_file_path
        self.num_channels = Params.audio_channels
        self.sampling_rate = Params.sampling_rate
        self.n_fft = Params.n_fft
        self.n_mels_bins = Params.n_mels_bins
        self.hop_length = Params.hop_length
        self.mel_wts = librosa.filters.mel(
            sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels_bins
        ).T
        self.time_masking = T.TimeMasking(time_mask_param=Params.time_mask)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Params.freq_mask)

    def load_data_from_csv(self) -> np.array:
        """
        Load the data from the csv file and return the data as a numpy array

        Args:
            None

        Returns:
            data_array: numpy array containing the data
        """
        data_df = pd.read_csv(self.csv_file_path, index_col=False, header=None)
        data_array = data_df.values
        return data_array

    def load_sproul_labels_and_preprocess(self) -> pd.DataFrame:
        """
        Load the Sproul labels and preprocess the data

        Args:
            None

        Returns:
            sproul_data: preprocessed Sproul data
        """
        sproul_data = pd.read_csv(self.sproul_text_file_path, sep="\t")

        sproul_data.columns = sproul_data.columns.str.strip()

        # Split the combined column into separate columns
        sproul_data[["Jday", "Time", "Duration", "Range(km)"]] = sproul_data[
            "Jday Time  Duration Range(km)"
        ].str.split(expand=True)

        # Drop the original combined column
        sproul_data = sproul_data.drop(columns=["Jday Time  Duration Range(km)"])

        # Convert 'Duration' and 'Range(km)' columns to appropriate data types
        sproul_data["Duration"] = sproul_data["Duration"].astype(int)
        sproul_data["Range(km)"] = sproul_data["Range(km)"].astype(float)

        return sproul_data

    def generate_metadata(
        self, num_spectrograms: int, spectrogram_duration: float
    ) -> pd.DataFrame:
        """
        Generate metadata for the dataset

        Args:
            num_spectrograms: number of spectrograms
            spectrogram_duration: duration of the spectrogram

        Returns:
            metadata: metadata for the dataset
        """
        metadata = pd.DataFrame(columns=["filename", "range_km", "fold", "target"])
        sproul_data = self.load_sproul_labels_and_preprocess()
        
        # 7.932 7.042  5.953 5.087 4.251 3.382 2.574 1.800 1.147 0.905 1.396 2.179
        test_ranges = [7.932, 7.042, 5.953, 5.087, 4.251, 3.382, 2.574, 1.800, 1.147, 0.905, 1.396, 2.179]
        
        # Iterate through each spectrogram
        for i in range(num_spectrograms):

            timestamp = i * spectrogram_duration

            # Convert the Duration column to seconds
            sproul_data["Duration_seconds"] = sproul_data["Duration"] * 60

            closest_idx = np.argmin(np.abs(sproul_data["Duration_seconds"] - timestamp))

            range_km = sproul_data.loc[closest_idx, "Range(km)"]

            filename = f"file_{i+1}.wav"
            # if Params.normal_split:
            fold = i % Params.total_folds + 1
            # else:
            #     if float(range_km) in test_ranges:
            #         fold = 6
            #     else:
            #         fold = i % 5 + 1
            target = float(range_km)

            new_row = pd.DataFrame(
                {
                    "filename": [filename],
                    "range_km": [range_km],
                    "fold": [fold],
                    "target": [target],
                }
            )
            metadata = pd.concat([metadata, new_row], ignore_index=True)

        # Check if folds are balanced
        fold_counts = metadata["fold"].value_counts()
        logging.info(fold_counts)

        return metadata

    def load_simulated_data_and_labels(self) -> tuple[np.array, np.array]:
        """
        Load the simulated data and labels

        Args:
            None

        Returns:
            data_array: numpy array containing the data
            labels: numpy array containing the labels
        """
        all_channels_data = []
        for i in range(Params.audio_channels):
            channel_data = joblib.load(
                f"{Params.simulated_time_serires_folder_path}/channel_{i+1}.pkl"
            )
            all_channels_data.append(channel_data)

        labels = joblib.load(Params.simulated_data_labels_path)
        all_channels_data = np.array(all_channels_data)
        # print(all_channels_data.shape)
        data_array = (
            all_channels_data.transpose((1, 0))
            if Params.data_format_mode == "time_series"
            else all_channels_data.transpose((1, 2, 0))
        )
        labels = np.array(labels)
        # print(data_array.shape)
        # print(labels.shape)
        # return data_array[:6000], labels[:6000]
        return data_array, labels

    def generate_metadata_for_simulated_data(
        self, num_spectrograms: int, labels: np.array
    ) -> pd.DataFrame:
        """
        Generate metadata for the simulated dataset

        Args:
            num_spectrograms: number of spectrograms
            labels: numpy array containing the labels

        Returns:
            metadata: metadata for the dataset
        """
        if (
            Params.data_format_mode != "time_series"
            and Params.data_format_mode != "chunked_list"
        ):
            logging.error(
                "Invalid data format mode. Accepted values are 'time_series' or 'chunked_list'."
            )
            raise ValueError

        # Create a new metadata DataFrame
        metadata = pd.DataFrame(columns=["filename", "range_km", "fold", "target"])

        if Params.data_format_mode == "time_series":
            logging.info("Using the time series format")
            # # Iterate through each spectrogram
            # for i in range(num_spectrograms):
            #     range_km = labels[i * self.sampling_rate]
            #     filename = f"file_{i+1}.wav"
            #     fold = i % Params.total_folds + 1
            #     target = float(range_km)

            #     new_row = pd.DataFrame(
            #         {
            #             "filename": [filename],
            #             "range_km": [range_km],
            #             "fold": [fold],
            #             "target": [target],
            #         }
            #     )
            #     metadata = pd.concat([metadata, new_row], ignore_index=True)
            # Shuffle indices to randomize the fold assignment
            indices = np.arange(num_spectrograms)
            np.random.shuffle(indices)

            # Assign folds randomly but evenly
            fold_assignments = (
                # indices % Params.total_folds + 1
                indices % 5 + 1
            )  # Folds will be 1 to total_folds
            # 7.932 7.042  5.953 5.087 4.251 3.382 2.574 1.800 1.147 0.905 1.396 2.179
            test_ranges = [7.932, 7.042, 5.953, 5.087, 4.251, 3.382, 2.574, 1.800, 1.147, 0.905, 1.396, 2.179]
            for i in range(num_spectrograms):
                range_km = labels[i * self.sampling_rate]
                filename = f"file_{i+1}.wav"
                if float(range_km) in test_ranges:
                    fold = 6
                else:
                    fold = fold_assignments[i]
                target = float(range_km)

                new_row = pd.DataFrame(
                    {
                        "filename": [filename],
                        "range_km": [range_km],
                        "fold": [fold],
                        "target": [target],
                    }
                )
                metadata = pd.concat([metadata, new_row], ignore_index=True)

        if Params.data_format_mode == "chunked_list":
            logging.info("Using the chunked list format")
            for i in range(num_spectrograms):
                range_km = labels[i]
                filename = f"file_{i+1}.wav"
                fold = i % Params.total_folds + 1
                target = float(range_km)

                new_row = pd.DataFrame(
                    {
                        "filename": [filename],
                        "range_km": [range_km],
                        "fold": [fold],
                        "target": [target],
                    }
                )
                metadata = pd.concat([metadata, new_row], ignore_index=True)

        # Check if folds are balanced
        fold_counts = metadata["fold"].value_counts()
        logging.info(fold_counts)

        return metadata

    def add_noise_with_snr(self, signal, snr_db):
        """
        Add Gaussian noise to a signal with a specified SNR (in dB).

        Parameters:
            signal (numpy array): Input signal
            snr_db (float): Desired Signal-to-Noise Ratio (SNR) in dB

        Returns:
            numpy array: Noisy signal
        """
        # Compute signal power
        signal_power = np.mean(signal**2)

        # Compute noise power based on desired SNR
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate Gaussian noise with calculated noise power
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

        # Add noise to the signal
        noisy_signal = signal + noise

        return noisy_signal

    def polarity_flip(self, signal):
        return -signal

    def time_warp_multichannel(self, signal, max_warp=0.2):
        """
        Time warp a multi-channel signal (time, channels).
        Warping is applied along the time axis for all channels.

        Parameters:
        - signal: np.array of shape (T, C)
        - max_warp: float, fraction of T to allow for warping shift

        Returns:
        - warped signal: same shape as input
        """
        T, C = signal.shape
        warp_point = np.random.randint(int(T * 0.3), int(T * 0.7))
        warp_amount = int(T * max_warp * (np.random.rand() * 2 - 1))  # ±max_warp shift

        # Create a warped time axis
        dst = np.arange(T)
        dst[warp_point:] = np.clip(dst[warp_point:] + warp_amount, 0, T - 1)

        # Interpolate each channel over the new warped timeline
        warped = np.zeros_like(signal)
        for ch in range(C):
            f = interp1d(
                dst,
                signal[:, ch],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            warped[:, ch] = f(np.arange(T))

        return warped

    def mixup_split_multichannel(self, signal, alpha=0.4, split_ratio=0.6):
        """
        Mixup between two parts of the same multi-channel signal (T, C).
        Ensures output has the same shape as input.
        """
        T, C = signal.shape
        split_idx = int(T * split_ratio)

        part1 = signal[:split_idx, :]
        part2 = signal[split_idx:, :]

        min_len = min(len(part1), len(part2))
        part1 = part1[:min_len]
        part2 = part2[:min_len]

        lam = np.random.beta(alpha, alpha)
        mixed = lam * part1 + (1 - lam) * part2

        # Reconstruct with padding to original length
        if split_ratio < 0.5:
            tail = signal[split_idx + min_len :, :]
            mixed_signal = np.concatenate([mixed, tail], axis=0)
        else:
            head = signal[: split_idx - min_len, :]
            mixed_signal = np.concatenate([head, mixed], axis=0)

        # Final check & pad if needed
        if mixed_signal.shape[0] < T:
            pad_len = T - mixed_signal.shape[0]
            padding = signal[-pad_len:, :]  # pad with trailing original data
            mixed_signal = np.concatenate([mixed_signal, padding], axis=0)
        elif mixed_signal.shape[0] > T:
            mixed_signal = mixed_signal[:T, :]

        return mixed_signal, lam

    def extract_features(
        self,
        data_array: np.array,
        metadata: pd.DataFrame,
        output_file_name: str,
    ) -> None:
        """
        Extract features from the data and save the dictionary to a pickle file

        Args:
            data_array: numpy array containing the data
            metadata: metadata for the dataset
            output_file_name: name of the output file

        Returns:
            None
        """
        data_dict = {}

        # if isinstance(self.sampling_rate, float) or "." in str(self.sampling_rate):
        #     logging.info("Sampling rate is a float...")
        #     self.sampling_rate = float(self.sampling_rate)
        #     total_seconds = len(data_array) // self.sampling_rate
        #     logging.info(f"total_seconds: {total_seconds}")
        #     num_slices = total_seconds // Params.spectrogram_duration

        #     # Compute the exact indices using cumulative rounding
        #     start_indices = np.round(np.arange(0, total_seconds) * self.sampling_rate).astype(int)
        #     end_indices = np.append(start_indices[1:], data_array.shape[0])  # Shifted indices for slicing

        #     for i in range(num_slices):
        #         filename = f"slice_{i}"
        #         range_km = metadata.iloc[i]["range_km"]

        #         start_idx, end_idx = start_indices[i], end_indices[i]

        #         data_dict[filename] = {
        #             "data": (
        #                 data_array[start_idx:end_idx]
        #                 if Params.data_format_mode == "time_series"
        #                 else data_array[i, :]
        #             ),
        #             "target": range_km,
        #         }

        # else:
        logging.info("Sampling rate is an integer...")
        self.sampling_rate = int(self.sampling_rate)
        for i in range(len(metadata)):
            filename = metadata.iloc[i]["filename"]
            range_km = metadata.iloc[i]["range_km"]
            start_idx = i * self.sampling_rate
            end_idx = min(
                (i + 1) * self.sampling_rate, data_array.shape[0]
            )  # Adjust for the last slice
            data_dict[filename] = {
                "data": (
                    data_array[start_idx:end_idx]
                    if Params.data_format_mode == "time_series"
                    else data_array[i, :]
                ),
                "target": range_km,
            }

        # self.sampling_rate = float(self.sampling_rate)
        # adjusted_sampling_rate = 3277 if self.sampling_rate == 3276.8 else int(self.sampling_rate)
        # logging.info(f"adjusted_sampling_rate: {adjusted_sampling_rate}")
        # time_correction_factor = 3276.8 / adjusted_sampling_rate

        # for i in range(len(metadata)):
        #     filename = metadata.iloc[i]["filename"]
        #     range_km = metadata.iloc[i]["range_km"]

        #     # Adjust indices by scaling back to original time reference
        #     start_idx = int(i * adjusted_sampling_rate * time_correction_factor)
        #     end_idx = min(
        #         int((i + 1) * adjusted_sampling_rate * time_correction_factor), data_array.shape[0]
        #     )  # Ensure we stay within bounds

        #     data_dict[filename] = {
        #         "data": (
        #             data_array[start_idx:end_idx]
        #             if Params.data_format_mode == "time_series"
        #             else data_array[i, :]
        #         ),
        #         "target": range_km,
        #     }

        audio_list = set(metadata["filename"])
        # output_dict = []
        output_dict = [[] for _ in range(Params.total_folds)]

        for index, row in metadata.iterrows():
            name = row["filename"]
            fold = row["fold"]
            target = row["target"]
            if name in audio_list:
                signal = data_dict[name]["data"]
                # if fold != 5:
                #     noise = np.random.normal(0, 0.01, signal.shape)
                #     signal = signal + noise
                if fold != 5 and fold != 6:
                    # signal_60 = self.add_noise_with_snr(signal, 2 * 60)
                    # signal_50 = self.add_noise_with_snr(signal, 2 * 50)
                    signal_40 = self.add_noise_with_snr(signal, 2 * 40)
                    signal_35 = self.add_noise_with_snr(signal, 2 * 35)
                    signal_30 = self.add_noise_with_snr(signal, 2 * 30)
                    signal_25 = self.add_noise_with_snr(signal, 2 * 25)
                    signal_20 = self.add_noise_with_snr(signal, 2 * 20)
                    signal_15 = self.add_noise_with_snr(signal, 2 * 15)
                    signal_10 = self.add_noise_with_snr(signal, 2 * 10)
                    # signal_warp = self.time_warp_multichannel(signal, max_warp=0.2)
                    # signal_flip = self.polarity_flip(signal)
                    # signal_mixup, _ = self.mixup_split_multichannel(
                    #     signal, alpha=0.3, split_ratio=0.7
                    # )

                target_value = data_dict[name]["target"]

                # # ensure all signal is of share (1500, 21):
                # assert (
                #     signal.shape == (self.sampling_rate, self.num_channels)
                #     and signal_warp.shape == (self.sampling_rate, self.num_channels)
                #     and signal_mixup.shape == (self.sampling_rate, self.num_channels)
                # ), f"Signal shape mismatch: signal={signal.shape}, signal_warp={signal_warp.shape}, signal_mixup={signal_mixup.shape}"

                # output_dict.append({
                #     "name": name,
                #     "target": float(target),
                #     "waveform": np.float32(signal),
                # })

                output_dict[int(fold) - 1].append(
                    {
                        "name": name,
                        "target": float(target),
                        "waveform": np.float32(signal),
                    }
                )
                # if fold != 5 and fold != 6:
                #     # output_dict[int(fold) - 1].append(
                #     #     {
                #     #         "name": name,
                #     #         "target": float(target),
                #     #         "waveform": np.float32(signal_60),
                #     #     }
                #     # )
                #     # output_dict[int(fold) - 1].append(
                #     #     {
                #     #         "name": name,
                #     #         "target": float(target),
                #     #         "waveform": np.float32(signal_50),
                #     #     }
                #     # )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_40),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_35),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_30),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_25),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_20),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_15),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_10),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_warp),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_flip),
                #         }
                #     )
                #     output_dict[int(fold) - 1].append(
                #         {
                #             "name": name,
                #             "target": float(target),
                #             "waveform": np.float32(signal_mixup),
                #         }
                #     )

                if index == 0:
                    logging.info("Logging the first audio file for sample check")
                    logging.info(
                        f"Processing {name}, Fold: {fold}, Target: {target_value}"
                    )
                    logging.info(f"waveform: {np.float32(signal)}")
                    logging.info(f"shape: {signal.shape}")
                    logging.info(f"target: {float(target)}")
                    logging.info("Continue to process the rest of the audio files...")

        with open(output_file_name, "wb") as f:
            pickle.dump(output_dict, f)
        logging.info("Data saving completed.")

    """
        output_dict = [[] for _ in range(Params.total_folds)]
        audio_list = set(metadata["filename"])

        for index, row in metadata.iterrows():
            name = row["filename"]
            fold = row["fold"]
            target = row["target"]
            if name in audio_list:
                signal, target_value = (
                    data_dict[name]["data"],
                    data_dict[name]["target"],
                )

                linear_spectra = self.get_spectrogram_from_array(
                    signal, augmentation=self.data_augmentation
                )

                mel_spectrograms = self.get_mel_spectrogram(linear_spectra)
                gcc_ph = self.get_gcc(linear_spectra)

                feat = np.concatenate((mel_spectrograms, gcc_ph), axis=-1)

                if np.isnan(feat).any():
                    logging.info("Feature extraction is generating nan outputs")
                    exit()

                output_dict[int(fold) - 1].append(
                    {
                        "name": name,
                        "target": float(target),
                        "waveform": np.float32(feat),
                    }
                )
                if index == 0:
                    logging.info("Logging the first audio file for sample check")
                    logging.info(
                        f"Processing {name}, Fold: {fold}, Target: {target_value}"
                    )
                    logging.info(f"waveform: {np.float32(feat)}")
                    logging.info(f"shape: {feat.shape}")
                    logging.info(f"target: {float(target)}")
                    logging.info("Continue to process the rest of the audio files...")

        logging.info("Length of the dataset: ")
        for i in range(Params.total_folds):
            logging.info(f"Fold {i+1}: {len(output_dict[i])}")

        # Save the dictionary using pickle
        logging.info(f"Saving the dictionary to {output_file_name}...")
        with open(output_file_name, "wb") as f:
            pickle.dump(output_dict, f)
        logging.info("Data saving completed.")

        # Use below as HLA south too heavy for 1 file
        # logging.info("Feature extraction completed. Saving training data...")
        # output_dict_train = output_dict[0] + output_dict[1] + output_dict[2] + output_dict[3]
        # with open("/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-1234-train.pkl", "wb") as f:
        #     pickle.dump(output_dict_train, f)

        # logging.info("Feature extraction completed. Saving validation data...")
        # output_dict_val = output_dict[4]
        # with open("/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-5-val.pkl", "wb") as f:
        #     pickle.dump(output_dict_val, f)

        # logging.info("Feature extraction completed. Saving test data...")
        # output_dict_test = output_dict[5]
        # with open("/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-6-test.pkl", "wb") as f:
        #     pickle.dump(output_dict_test, f)

        # logging.info("Data saving completed.")
    """

    def nCr(self, n: int, r: int) -> int:
        return math.factorial(n) // math.factorial(r) // math.factorial(n - r)

    def get_gcc(self, linear_spectra: np.array) -> np.array:
        """
        Compute the Generalized Cross-Correlation (GCC) from the linear spectrogram

        Args:
            linear_spectra: linear spectrogram

        Returns:
            gcc_feat: GCC feature
        """
        gcc_channels = self.nCr(linear_spectra.shape[-1], 2)
        # logging.info("gcc_channels: ", gcc_channels)
        gcc_feat = np.zeros((linear_spectra.shape[0], self.n_mels_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m + 1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.0j * np.angle(R)))
                cc = np.concatenate(
                    (cc[:, -self.n_mels_bins // 2 :], cc[:, : self.n_mels_bins // 2]),
                    axis=-1,
                )
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def add_noise(self, signal: np.array, snr_db: int) -> np.array:
        """
        Adds Gaussian noise to a signal to achieve a specified SNR.

        Args:
            signal: Input signal (numpy array or torch tensor)
            snr_db: Desired Signal-to-Noise Ratio in dB

        Returns:
            Noisy signal
        """

        signal_power = np.mean(signal**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

        noisy_signal = signal + noise

        return noisy_signal

    def time_shift(self, signal: np.array, shift: int) -> np.array:
        """
        Shift the signal in time by the specified number of frames

        Args:
            signal: input signal
            shift: number of frames to shift the signal by

        Returns:
            shifted signal
        """
        return np.roll(signal, shift)

    def get_spectrogram_from_array(self, audio, augmentation=None) -> np.array:
        """
        Call the get_spectrogram function to get linear spectrogram from the audio input
        with the specified augmentation and number of frames

        Args:
            audio: audio input
            augmentation: augmentation type

        Returns:
            audio_spec: linear spectrogram
        """
        nb_feat_frames = int(len(audio) / float(self.hop_length))

        audio_spec = self.get_spectrogram(audio, nb_feat_frames, augmentation)
        return audio_spec

    def get_spectrogram(
        self, audio_input: np.array, nb_frames, augmentation=None
    ) -> np.array:
        """
        Compute the linear spectrogram

        Args:
            audio_input: audio input
            nb_frames: number of frames
            augmentation: augmentation type

        Returns:
            spectra: linear spectrogram
        """
        n_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(n_ch):
            signal = audio_input[:, ch_cnt].astype(float)

            if isinstance(augmentation, int) and augmentation != 0:
                signal = self.add_noise(signal, augmentation)

            stft_ch = librosa.core.stft(
                np.asfortranarray(signal),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=2 * self.hop_length,
                window="hann",
            )

            if augmentation == "time_masking" or augmentation == "freq_masking":
                magnitude, phase = librosa.magphase(stft_ch)
                magnitude = torch.from_numpy(magnitude)

                if augmentation == "time_masking":
                    magnitude = self.time_masking(magnitude)
                elif augmentation == "freq_masking":
                    magnitude = self.freq_masking(magnitude)

                magnitude = magnitude.cpu().numpy()
                stft_ch = magnitude * np.exp(1j * phase)

            spectra.append(stft_ch[:, :nb_frames])
        return np.array(spectra).T

    def get_mel_spectrogram(self, linear_spectra: np.array) -> np.array:
        """
        Compute the mel spectrogram from the linear spectrogram

        Args:
            linear_spectra: linear spectrogram

        Returns:
            mel_feat: mel spectrogram
        """
        mel_feat = np.zeros(
            (linear_spectra.shape[0], self.n_mels_bins, linear_spectra.shape[-1])
        )
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
            mel_spectra = np.dot(mag_spectra, self.mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat
