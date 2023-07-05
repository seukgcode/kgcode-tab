import json
from pathlib import Path
from typing import Any

import orjson

from . import PathLike


def read_json(path: PathLike) -> Any:
    return orjson.loads(Path(path).read_text(encoding="utf8"))


def write_json(obj: Any, path: PathLike, **kwargs) -> int:
    return Path(path).write_text(json.dumps(obj, **kwargs), encoding="utf8")