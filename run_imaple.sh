#!/bin/bash

# Description: This script extracts features from the dataset and saves them to a pickle file.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2024-10-16
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# @@@

show_help() {
    echo "Usage: ./run.sh {preprocess_features|train_model|test_model|activate_venv|help}"
    echo ""
    echo "SA-Net: Underwater Sound Source Localization"
    echo ""
    echo "Options:"
    echo ""
    echo "  preprocess_features  - Start the feature extraction"
    echo "                      Command: ./run.sh preprocess_features"
    echo ""
    echo "  train_model          - Train mode (test will automatically run with the option \`run_inference_mode=True\` in \`parameters.yaml\`)"
    echo "                      Command: ./run.sh train_model"
    echo ""
    echo "  test_model           - Test mode"
    echo "                      Command: ./run.sh test_model"
    echo ""
    echo "  help                 - Show this help message"
    echo "                      Command: ./run.sh help"
    echo ""
}

if [ -z "$1" ] || [ "$1" == "help" ]; then
    show_help
    exit 0
fi

# need to activate the virtual environment

if [ "$1" == "activate_venv" ]; then
    source swell-env/bin/activate
    echo "Virtual environment activated"
    exit 0
elif [ "$1" == "preprocess_features" ]; then
    python3 underwater-ssl/main.py --params-file=parameters.yaml feature_extractor
elif [ "$1" == "train_model" ]; then
    python3 underwater-ssl/main.py --params-file=parameters.yaml model_trainer
elif [ "$1" == "test_model" ]; then
    python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only model_trainer
else
    echo "Invalid option: $1"
    show_help
    exit 1
fi
