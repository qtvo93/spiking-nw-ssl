# Spiking-NN-SSL


## How to run

## If you have Docker installed on your machine, you can run the code with docker as well.
### Build the image if you run it for the first time
```bash
./run.sh build_docker_image
```

## If you don't have Docker installed on your machine, remove the option `--with-docker` and run app as normally.
### To start the feature extraction
```bash
./run.sh preprocess_features --with-docker
```

### To train (test will automatically run with the option `run_inference_mode=True` in `parameters.yaml` file.)
```bash
./run.sh train_model --with-docker
```

### To test your pretrained model, simply set the `pretrained_model_path` to your model and run
```bash
./run.sh test_model --with-docker
```

## For iMaPLe Servers
### Please run it in your own .env 
### OR Don't override my virtual env if you run as below.
```bash
cd ~qv23/user_data/clean-code-swell
```

```bash
source swell-env/bin/activate
```

## Before running any scripts, make sure you check the `parameters.yaml` for file path, data, and other hyper-parameters, etc
### To start the feature extraction
```bash
python3 underwater-ssl/process_features.py --params-file=parameters.yaml
```

### To train (test will automatically run with the option `run_inference_mode=True` in `parameters.yaml` file.)
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml
```

### To test your pretrained model, simply set the `pretrained_model_path` to your model and run
```bash
python3 underwater-ssl/main.py --params-file=parameters.yaml --run-test-only
```

