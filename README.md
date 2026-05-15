# Spiking-NN-SSL

## This codebase was developed as part of the experiments for the following papers
Codebase Author: Quoc Thinh Vo - qv23 [at] drexel [dot] edu

If you find this code useful and use it, in whole or in part, please consider citing the following papers:

#### [Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation](https://ieeexplore.ieee.org/abstract/document/11226378/)
Published in 2026 51st IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
```
@inproceedings{vo2026sa-net,
  title={Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation},
  author={Vo, Quoc Thinh and Han, David K},
  booktitle={2026 51st IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  pages={1--5},
  year={2026},
  organization={IEEE}
}
```

#### [Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/11226378/)
Published in 2025 33rd European Signal Processing Conference (EUSIPCO)
```
@inproceedings{vo2025aca-net,
  title={Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation},
  author={Vo, Quoc Thinh and Woods, Joe and Chowdhury, Priontu and Han, David K},
  booktitle={2025 33rd European Signal Processing Conference (EUSIPCO)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## How to Run

Before running scripts, check `parameters.yaml` for paths, data settings, and hyperparameters.

### Option 1: Run with UV

Prerequesite: https://docs.astral.sh/uv/

Simply run any scripts with `uv`

```bash
uv sync
uv run python3 underwater-ssl/main.py --params-file=parameters.yaml feature_extractor
uv run python3 underwater-ssl/main.py --params-file=parameters.yaml model_trainer
uv run python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only model_trainer
```

### Option 2: Run with Docker

#### Build the image (first-time setup)
```bash
./run.sh build_docker_image
```

#### Start feature extraction (for the raw-signal spiking setup, this step simply splits the audio into 1-second segments and assigns labels accordingly)
```bash
./run.sh preprocess_features --with-docker
```

#### Train the model
Test will automatically run when `run_inference_mode=True` in `parameters.yaml`.
```bash
./run.sh train_model --with-docker
```

#### Test a pretrained model
Set `pretrained_model_path` in `parameters.yaml`, then run:
```bash
./run.sh test_model --with-docker
```

### Option 3: Run without Docker

If `uv` and `Docker` are not available, remove the `--with-docker` option and run the Python scripts directly.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh preprocess_features
./run.sh train_model
./run.sh test_model
```

### Option 4: Direct Python Execution

#### Init virtual env
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install the dependencies
```bash
pip install -r requirements.txt
```
#### Start feature extraction
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml feature_extractor
```

#### Train the model
Test will automatically run when `run_inference_mode=True` in `parameters.yaml`.
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml model_trainer
```

#### Test a pretrained model
Set `pretrained_model_path` in `parameters.yaml`, then run:
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only model_trainer
```

### For iMaPLe Research Lab Servers

Please run in your own environment.

Do not override my virtual environment if you use the setup below:
```bash
cd ~qv23/user_data/clean-code-swell
```

```bash
source swell-env/bin/activate
```
### Run with ./run_imaple.sh
```bash
./run_imaple.sh preprocess_features
./run_imaple.sh train_model
./run_imaple.sh test_model
```
