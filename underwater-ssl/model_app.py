# Description: This file is the main entry point for training and inference
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import os
import pickle
import logging
import sys
import argparse
import torch
import yaml
import wandb
import time
import torch.nn as nn
import torch.optim as optim

from sa_net.inference import Inference
from sa_net.model import SA_NET
from sa_net.trainer import Trainer
from loss.loss import Loss
from utils.data_loader import CustomDataLoader
from utils.parameters import Params


class MainApp(object):
    def __init__(self):
        super().__init__()
        self.model = SA_NET()
        self.loss = Loss()
        self.save_dir = Params.save_dir
        self.best_model_path = Params.best_model_path
        self.device = Params.device
        self.learning_rate = float(Params.learning_rate)
        self.num_epochs = Params.num_epochs
        self.parallel_mode = Params.parallel_mode

    def load_data_set(self) -> tuple:
        """
        Load the training, validation, and test datasets

        Args:
            None

        Returns:
            train_loader: training data loader
            val_loader: validation data loader
            test_loader: test data loader
        """
        dataset_loader = CustomDataLoader()
        train_loader, val_loader, test_loader = dataset_loader.load_data()
        return train_loader, val_loader, test_loader

    def init_model(self, parallel: bool = True) -> SA_NET:
        """
        Initialize the model and move it to the device

        Args:
            None

        Returns:
            model: initialized model
        """
        model = self.model
        model = model.to(self.device)
        if parallel and torch.cuda.device_count() > 1:
            logging.info(f"USE ALL {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total parameters: {total_params}")

        return model

    def train(self, train_loader, val_loader) -> None:
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

        os.makedirs(self.save_dir, exist_ok=True)
        best_model_path = os.path.join(self.save_dir, self.best_model_path)

        logging.info(f"Using device: {self.device}")
        model = self.init_model(self.parallel_mode)

        if Params.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer not supported")

        if Params.finetune_mode:
            model.load_state_dict(
                torch.load(Params.pretrained_model_path), strict=False
            )
            logging.info("Finetuning model...")

        Trainer(
            best_model_path,
            optimizer,
            self.loss,
            self.num_epochs,
            self.device,
            model,
            train_loader,
            val_loader,
        ).train_supervised_model()

    def inference(self, test_loader) -> None:
        """
        Perform inference on the test dataset using the trained model

        Args:
            model: trained model
            test_loader: test data loader
            best_model_path: path to the best model checkpoint

        Returns:
            None
        """
        model = self.init_model(self.parallel_mode)

        Inference(model, test_loader).inference()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    if Params.log_verbose:
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        for handler in log.handlers:
            handler.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Run the main program with parameters from a YAML file."
    )
    parser.add_argument(
        "--params-file",
        type=str,
        required=True,
        help="Path to the parameters YAML file.",
    )
    parser.add_argument(
        "--run-test-only",
        action="store_true",
        help="Run the program in inference mode only.",
    )
    args = parser.parse_args()

    start_time = time.time()
    # Load parameters from YAML file
    try:
        params_file = args.params_file
        Params.load_from_yaml(params_file)
    except Exception as e:
        logging.error(f"Error loading parameters from YAML file: {e}")
        raise e

    if Params.run_with_wandb:
        wandb.init(
            project=Params.wandb_training_project, name=Params.wandb_training_name
        )
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
        wandb.config.update(params)

        artifact = wandb.Artifact("run_parameters", type="config")
        artifact.add_file(params_file)
        wandb.log_artifact(artifact)

    main = MainApp()
    train_loader, val_loader, test_loader = main.load_data_set()

    if args.run_test_only:
        logging.info("Running inference mode...")
        main.inference(test_loader)
        logging.info("Inference completed!")
        sys.exit(0)

    logging.info("Start training the model...")
    main.train(train_loader, val_loader)
    logging.info("Training completed!")
    if Params.run_inference_mode == True:
        logging.info("Running test on the test dataset...")
        main.inference(test_loader)
        logging.info("Inference completed!")

    wandb.finish() if Params.run_with_wandb else None
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")
