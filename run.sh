#!/bin/bash

# Description: This script extracts features from the dataset and saves them to a pickle file.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-05
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

show_help() {
    echo "Usage: ./run.sh {preprocess_features|train_model|test_model|build_docker_image|help} [options]"
    echo ""
    echo "SA-Net: Underwater Sound Source Localization"
    echo ""
    echo "Options:"
    echo ""
    echo "  preprocess_features  - Start the feature extraction"
    echo "                      Command: ./run.sh preprocess_features [--with-docker]"
    echo ""
    echo "  train_model          - Train mode (test will automatically run with the option \`run_inference_mode=True\` in \`parameters.yaml\`)"
    echo "                      Command: ./run.sh train_model [--with-docker]"
    echo ""
    echo "  test_model           - Test mode"
    echo "                      Command: ./run.sh test_model [--with-docker]"
    echo ""
    echo "  help                 - Show this help message"
    echo "                      Command: ./run.sh help"
    echo ""
}

if [ -z "$1" ] || [ "$1" == "help" ]; then
    show_help
    exit 0
fi

WITH_DOCKER=false
if [ "$2" == "--with-docker" ]; then
    WITH_DOCKER=true
fi


if [ "$1" == "build_docker_image" ]; then
    docker build -t SA-Net-swellex96:v1 .
    exit 0
fi

if [ "$1" == "preprocess_features" ]; then
    if [ "$WITH_DOCKER" == true ]; then
        docker run -it --rm -v $(pwd)/parameters.yaml:/app/parameters.yaml SA-Net-swellex96:v1 bash -c "uv run python3 underwater-ssl/main.py --params-file=/app/parameters.yaml feature_extractor"
    else
        uv run python3 underwater-ssl/main.py --params-file=parameters.yaml feature_extractor
    fi
elif [ "$1" == "train_model" ]; then
    if [ "$WITH_DOCKER" == true ]; then
        docker run -it --rm -v $(pwd)/parameters.yaml:/app/parameters.yaml SA-Net-swellex96:v1 bash -c "uv run python3 underwater-ssl/main.py --params-file=/app/parameters.yaml model_trainer"
    else
        uv run python3 underwater-ssl/main.py --params-file=parameters.yaml model_trainer
    fi
elif [ "$1" == "test_model" ]; then
    if [ "$WITH_DOCKER" == true ]; then
        docker run -it --rm -v $(pwd)/parameters.yaml:/app/parameters.yaml SA-Net-swellex96:v1 bash -c "uv run python3 underwater-ssl/main.py --params-file=/app/parameters.yaml --run-test-only model_trainer"
    else
        uv run python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only model_trainer
    fi
else
    echo "Invalid option: $1"
    show_help
    exit 1
fi
