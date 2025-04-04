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
