# Description: This script extracts features from the dataset and saves them to a pickle file.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

import logging
import sys
import argparse

from utils.feature_extraction import FeatureExtraction
from utils.parameters import Params


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
        description="Run the preprocessing features program with parameters from a YAML file."
    )
    parser.add_argument(
        "--params-file",
        type=str,
        required=True,
        help="Path to the parameters YAML file.",
    )
    args = parser.parse_args()

    # Load parameters from YAML file
    try:
        params_file = args.params_file
        Params.load_from_yaml(params_file)
    except Exception as e:
        logging.error(f"Error loading parameters from YAML file: {e}")
        raise e

    logging.info("Extracting features from the dataset...")

    feature_extraction = FeatureExtraction(data_augmentation=Params.data_augmentation)

    if Params.simulated_data_mode:
        logging.info("Using simulated data...")
        data_array, labels = feature_extraction.load_simulated_data_and_labels()
        if Params.simulated_num_spectrograms:
            num_spectrograms = Params.simulated_num_spectrograms
        else:
            num_spectrograms = int(
                len(data_array) / (Params.sampling_rate * Params.sample_duration)
            )
        metadata = feature_extraction.generate_metadata_for_simulated_data(
            num_spectrograms=num_spectrograms,
            labels=labels,
        )
    else:
        logging.info("Using real data...")
        data_array = feature_extraction.load_data_from_csv()
        sproul_data = feature_extraction.load_sproul_labels_and_preprocess()

        num_spectrograms = int(
            len(data_array) / (Params.sampling_rate * Params.sample_duration)
        )
        metadata = feature_extraction.generate_metadata(
            num_spectrograms=num_spectrograms,
            sample_duration=Params.sample_duration,
        )

    feature_extraction.extract_features(
        data_array=data_array,
        metadata=metadata,
        output_file_name=Params.dataset_path,
    )
    logging.info("Feature extraction completed.")
