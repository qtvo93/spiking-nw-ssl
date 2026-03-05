# Description: Custom dataset class for loading data
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-05
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

import pickle
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.parameters import Params
from utils.set_seed import SetSeed


class CustomDataset(Dataset):
    def __init__(self, data):
        self.set_seed = SetSeed(seed=42)
        self.set_seed.set_seed()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]["waveform"]
        target = self.data[idx]["target"]
        return data, target


class CustomDataLoader(object):
    def __init__(self):
        self.set_seed = SetSeed(seed=42)
        self.set_seed.set_seed()

    def load_data(self) -> tuple:
        """
        Load the training, validation, and test datasets

        Args:
            None

        Returns:
            train_loader: training data loader
            val_loader: validation data loader
            test_loader: test data loader
        """
        train_data, val_data, test_data = [], [], []
        if Params.multiple_datasets_mode:
            logging.info("Multiple datasets mode is enabled!")
            for dataset in Params.train_datasets:
                logging.info(f"Loading data from {dataset}...")
                with open(dataset, "rb") as f:
                    training_data_dict = pickle.load(f)
                    train_data.extend(training_data_dict[0])
            for dataset in Params.validation_datasets:
                logging.info(f"Loading data from {dataset}...")
                with open(dataset, "rb") as f:
                    val_data_dict = pickle.load(f)
                    val_data.extend(val_data_dict[0])
            for dataset in Params.test_datasets:
                logging.info(f"Loading data from {dataset}...")
                with open(dataset, "rb") as f:
                    test_data_dict = pickle.load(f)
                    test_data.extend(test_data_dict[0])
        else:
            logging.info("Multiple datasets mode is disabled!")
            logging.info("Loading data in...")
            # hard code this to load the heavy data
            regular_file = True
            if regular_file:
                with open(Params.dataset_path, "rb") as f:
                    training_data_dict = pickle.load(f)

                if Params.cross_validation_mode:
                    for fold in Params.train_folds:
                        train_data.extend(training_data_dict[fold])
                    for fold in Params.validation_folds:
                        val_data.extend(training_data_dict[fold])
                    for fold in Params.test_folds:
                        test_data.extend(training_data_dict[fold])
                else:
                    # all_data = [
                    #     item
                    #     for sublist in training_data_dict[: Params.total_folds]
                    #     for item in sublist
                    # ]

                    all_data = (
                        training_data_dict[0]
                        + training_data_dict[1]
                        + training_data_dict[2]
                        + training_data_dict[3]
                        + training_data_dict[4]
                        + training_data_dict[5]
                    )

                    # # hard code this to load the right - receding data only
                    # all_data = training_data_dict[5]
                    all_train_data, test_data = train_test_split(
                        all_data, test_size=Params.test_size, random_state=42
                    )
                    train_data, val_data = train_test_split(
                        all_train_data, test_size=0.15, random_state=42
                    )
            else:
                with open(
                    "/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-1234-train.pkl",
                    "rb",
                ) as f:
                    training_data_dict = pickle.load(f)

                with open(
                    "/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-5-val.pkl",
                    "rb",
                ) as f:
                    val_data_dict = pickle.load(f)

                with open(
                    "/mnt/active_storage/qv23/DCASE2024/swell24/swellex-data-HLA-South-6-1sec-6-test.pkl",
                    "rb",
                ) as f:
                    test_data_dict = pickle.load(f)

                train_data = training_data_dict
                val_data = val_data_dict
                test_data = test_data_dict
            # # short cut to test on real data now
            # val_data, test_data = [], []
            # logging.info(f"Loading real data now...")
            # with open("/mnt/active_storage/qv23/DCASE2024/swell24/swellex-6folds-real.pkl", "rb") as f:
            #     real_data = pickle.load(f)
            # for fold in [0, 1, 2, 3, 4]:
            #     val_data.extend(real_data[fold])
            # for fold in [5]:
            #     test_data.extend(real_data[fold])

            train_dataset = CustomDataset(data=train_data)
            train_loader = DataLoader(
                train_dataset, batch_size=Params.batch_size, shuffle=True
            )

            val_dataset = CustomDataset(data=val_data)
            val_loader = DataLoader(
                val_dataset, batch_size=Params.batch_size, shuffle=False
            )

            test_dataset = CustomDataset(data=test_data)
            test_loader = DataLoader(
                test_dataset, batch_size=Params.batch_size, shuffle=False
            )

        logging.info(f"Number of training samples: {len(train_data)}")
        logging.info(f"Number of validation samples: {len(val_data)}")
        logging.info(f"Number of test samples: {len(test_data)}")
        logging.info("Data loaded successfully!")

        return train_loader, val_loader, test_loader
