# Spiking-NN-SSL

## This codebase was developed as part of the experiments for the following papers

Author: Quoc Thinh Vo - qv23 [at] drexel [dot] edu

If you find this code useful and use it, in whole or in part, please consider citing the following papers:
@@@

## How to Run

### Option 1: Run with Docker

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

### Option 2: Run without Docker

If Docker is not available, remove the `--with-docker` option and run the Python scripts directly.

### For iMaPLe Servers

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

Or simply run with `uv`
```bash
uv sync
uv run python3 scripts/
```

### Direct Python Execution

Before running scripts, check `parameters.yaml` for paths, data settings, and hyperparameters.

#### Start feature extraction
```bash
python3 underwater-ssl/process_features.py --params-file=parameters.yaml
```

#### Train the model
Test will automatically run when `run_inference_mode=True` in `parameters.yaml`.
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml
```

#### Test a pretrained model
Set `pretrained_model_path` in `parameters.yaml`, then run:
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only
```
