import sys
import typer
import logging
import wandb
import yaml
import time
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from options import Options, BaseOptions
from model_app import MainApp
from utils.feature_extraction import FeatureExtraction
from utils.parameters import Params

app = typer.Typer(
    help="Run the preprocessing features program or model Trainer with parameters from a YAML file."
)


@app.callback()
def main(
    params_file: str = typer.Option(
        BaseOptions().params_file,
        "--params-file",
        "-p",
        help="Path to the parameters YAML file.",
    ),
    run_test_only: bool = typer.Option(
        BaseOptions().run_test_only,
        "--run-test-only",
        "-t",
        help="Run the program in inference mode only.",
    ),
):
    Options(params_file=params_file, run_test_only=run_test_only)

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


@app.command(
    "model_trainer", help="Run the model Trainer with parameters from a YAML file."
)
def model_trainer():
    logging.info("Start the model trainer app...")

    common_options = Options()
    params_file = common_options.params_file
    run_test_only = common_options.run_test_only

    start_time = time.time()
    # Load parameters from YAML file
    try:
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

    model_trainer = MainApp()
    train_loader, val_loader, test_loader = model_trainer.load_data_set()

    if run_test_only:
        logging.info("Running inference mode...")
        model_trainer.inference(test_loader)
        logging.info("Inference completed!")
        sys.exit(0)

    logging.info("Start training the model...")
    model_trainer.train(train_loader, val_loader)
    logging.info("Training completed!")
    if Params.run_inference_mode == True:
        infer_time = time.time()
        logging.info("Running test on the test dataset...")
        model_trainer.inference(test_loader)
        logging.info("Inference completed!")
        logging.info(f"INFER: {time.time() - infer_time}")

    wandb.finish() if Params.run_with_wandb else None
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")


@app.command(
    "feature_extractor",
    help="Run the preprocessing features program with parameters from a YAML file.",
)
def feature_extractor():
    logging.info("Start the feature extraction app...")

    common_options = Options()
    params_file = common_options.params_file

    start_time = time.time()
    # Load parameters from YAML file
    try:
        Params.load_from_yaml(params_file)
    except Exception as e:
        logging.error(f"Error loading parameters from YAML file: {e}")
        raise e

    logging.info("Extracting features from the dataset...")

    feature_extraction = FeatureExtraction(data_augmentation=Params.data_augmentation)

    # check if Params.dataset_path contains correct folder path
    parent_folder = "/".join(Params.dataset_path.split("/")[:-1])
    if not os.path.exists(parent_folder):
        logging.info(f"Folder does not exist. Creating folder: {parent_folder}")
        os.makedirs(parent_folder)

    if Params.simulated_data_mode:
        logging.info("Using simulated data...")
        data_array, labels = feature_extraction.load_bell_simulated_data_and_labels()
        if Params.simulated_num_samples:
            num_spectrograms = Params.simulated_num_samples
        else:
            num_spectrograms = int(
                len(data_array) / (Params.sampling_rate * Params.sample_duration)
            )
        # metadata = feature_extraction.generate_bell_metadata_for_simulated_data(
        #     num_spectrograms=num_spectrograms,
        #     labels=labels,
        # )
        metadata = feature_extraction.generate_bell_metadata_for_simulated_data()
    else:
        logging.info("Using real data...")
        data_array = feature_extraction.load_data_from_csv()

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
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    app()
