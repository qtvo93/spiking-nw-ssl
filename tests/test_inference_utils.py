from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as app_main


def test_feature_extractor_main_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeOptions:
        def __init__(
            self, params_file: str = "parameters.yaml", run_test_only: bool = False
        ):
            self.params_file = params_file
            self.run_test_only = run_test_only

    class FakeFeatureExtraction:
        def __init__(self, data_augmentation: str):
            calls["data_augmentation"] = data_augmentation

        def load_data_from_csv(self) -> np.ndarray:
            calls["load_data_from_csv"] = True
            return np.zeros((20, 2), dtype=np.float32)

        def generate_metadata(
            self, num_samples: int, sample_duration: float
        ) -> pd.DataFrame:
            calls["generate_metadata"] = (num_samples, sample_duration)
            return pd.DataFrame(
                {
                    "filename": ["a", "b"],
                    "range_km": [1.0, 2.0],
                    "fold": [1, 2],
                    "target": [1.0, 2.0],
                }
            )

        def extract_features(self, data_array, metadata, output_file_name: str) -> None:
            calls["extract_features"] = output_file_name

    output_file = tmp_path / "generated" / "dataset.pkl"

    monkeypatch.setattr(app_main, "Options", FakeOptions)
    monkeypatch.setattr(app_main, "FeatureExtraction", FakeFeatureExtraction)
    monkeypatch.setattr(app_main.Params, "load_from_yaml", lambda _: None)
    monkeypatch.setattr(app_main.Params, "data_augmentation", "none")
    monkeypatch.setattr(app_main.Params, "dataset_path", str(output_file))
    monkeypatch.setattr(app_main.Params, "simulated_data_mode", False)
    monkeypatch.setattr(app_main.Params, "sampling_rate", 10)
    monkeypatch.setattr(app_main.Params, "sample_duration", 1.0)

    app_main.feature_extractor()

    assert calls["data_augmentation"] == "none"
    assert calls["load_data_from_csv"] is True
    assert calls["generate_metadata"] == (2, 1.0)
    assert calls["extract_features"] == str(output_file)


def test_model_trainer_main_flow_runs_train_then_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"inference_count": 0}

    class FakeOptions:
        def __init__(
            self, params_file: str = "parameters.yaml", run_test_only: bool = False
        ):
            self.params_file = params_file
            self.run_test_only = run_test_only

    class FakeMainApp:
        def load_data_set(self):
            return "train_loader", "val_loader", "test_loader"

        def train(self, train_loader, val_loader) -> None:
            calls["train"] = (train_loader, val_loader)

        def inference(self, test_loader) -> None:
            calls["inference_count"] = int(calls["inference_count"]) + 1
            calls["inference_arg"] = test_loader

    monkeypatch.setattr(app_main, "Options", FakeOptions)
    monkeypatch.setattr(app_main, "MainApp", FakeMainApp)
    monkeypatch.setattr(app_main.Params, "load_from_yaml", lambda _: None)
    monkeypatch.setattr(app_main.Params, "run_with_wandb", False)
    monkeypatch.setattr(app_main.Params, "run_inference_mode", True)

    app_main.model_trainer()

    assert calls["train"] == ("train_loader", "val_loader")
    assert calls["inference_count"] == 1
    assert calls["inference_arg"] == "test_loader"


def test_model_trainer_test_only_runs_inference_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {"train_called": False, "inference_count": 0}

    class FakeOptions:
        def __init__(
            self, params_file: str = "parameters.yaml", run_test_only: bool = False
        ):
            self.params_file = params_file
            self.run_test_only = True

    class FakeMainApp:
        def load_data_set(self):
            return "train_loader", "val_loader", "test_loader"

        def train(self, train_loader, val_loader) -> None:
            calls["train_called"] = True

        def inference(self, test_loader) -> None:
            calls["inference_count"] = int(calls["inference_count"]) + 1

    monkeypatch.setattr(app_main, "Options", FakeOptions)
    monkeypatch.setattr(app_main, "MainApp", FakeMainApp)
    monkeypatch.setattr(app_main.Params, "load_from_yaml", lambda _: None)
    monkeypatch.setattr(app_main.Params, "run_with_wandb", False)
    monkeypatch.setattr(app_main.Params, "run_inference_mode", True)

    with pytest.raises(SystemExit) as raised:
        app_main.model_trainer()

    assert raised.value.code == 0
    assert calls["train_called"] is False
    assert calls["inference_count"] == 1
