# Description: This file contains the dataclasses for the parameters used in the model training and inference.
# Author: Quoc Thinh Vo - qv23@drexel.edu
# Last Modified: 2026-03-20
# If you refer to or use this code, in whole or in part, please consider citing the following papers:
# 1. Spiking Attention Network: A Hybrid Neuromorphic Approach to Underwater Acoustic Localization and Zero-shot Adaptation
# 2. Adaptive Control Attention Network for Underwater Acoustic Localization and Domain Adaptation

from dataclasses import dataclass
from typing import Any, ClassVar, Dict


class Singleton(type):
    _instances: ClassVar[Dict[Any, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> type:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class BaseOptions:
    params_file: str = ""
    run_test_only: bool = False


@dataclass
class Options(BaseOptions, metaclass=Singleton):
    pass
