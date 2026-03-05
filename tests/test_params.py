from pathlib import Path

import torch
import yaml

from utils.parameters import Params


def test_load_from_yaml_normalizes_device(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump({"device": "cpu"}))

    old_device = Params.device
    try:
        Params.load_from_yaml(str(params_file))
        assert isinstance(Params.device, torch.device)
        assert Params.device.type == "cpu"
    finally:
        Params.device = old_device
