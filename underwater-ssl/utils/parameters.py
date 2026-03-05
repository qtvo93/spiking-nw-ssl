# Description: This file contains the parameters used in the model training and inference.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-05
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

import torch
import yaml


class Params(object):
    wandb_training_project = "sa-net"
    wandb_traning_name = "swellex-run-no-1"
    run_with_wandb = False
    batch_size = 32
    sample_duration = 1.0
    data_format_mode = "time_series"
    audio_channels = 21
    learning_rate = 1e-4
    num_samples = 4500
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = "Adam"
    sampling_rate = 200
    multiple_datasets_mode = False
    train_datasets = []
    validation_datasets = []
    test_datasets = []
    cross_validation_mode = True
    total_folds = 6
    train_folds = [1]
    validation_folds = [1]
    test_folds = [0, 2, 3, 4, 5]
    test_size = 0.2
    run_inference_mode = True
    finetune_mode = True  # (If True, pretrained_model_path must be provided)
    log_verbose = False
    print_first_10th_predictions = False
    prediction_output_image_plot = "true_vs_predicted_labels.png"
    csv_file_path = "underwater-ssl/data/s5/vla/vla_raw.csv"
    sproul_text_file_path = "underwater-ssl/data/s5/vla/SproulToVLA.S5.txt"
    best_model_path = "best_model_2-simu.pth"
    parallel_mode = False
    ######################
    #   Simulated data   #
    ######################
    # This is for further experiments with simulated data, not used in the main paper of SA-NET
    simulated_data_mode = False
    simulated_num_samples = 4500
    simulated_data_labels_path = "/mnt/researchfiles/cluster_data/archive/underwater_acoustics/Simulated_Datasets/Scenario_1/Chunked_Time_Series/dataset_1/labels/range_labels.pkl"
    simulated_time_serires_folder_path = "/mnt/researchfiles/cluster_data/archive/underwater_acoustics/Simulated_Datasets/Scenario_1/Chunked_Time_Series/dataset_1/features"
    #######################
    # active storage path #
    #######################
    pretrained_model_path = (
        "/mnt/active_storage/swell24/model_checkpoints/best_model_1-simu.pth"
    )
    dataset_path = "/mnt/active_storage/swell24/swellex-6folds-real.pkl"
    save_dir = "/mnt/active_storage/swell24/model_checkpoints"

    @classmethod
    def load_from_yaml(cls, params_file):
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
        for key, value in params.items():
            setattr(cls, key, value)

        # safety check for cuda availability
        cls.device = cls.device if torch.cuda.is_available() else "cpu"
