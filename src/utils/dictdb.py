import time
from pathlib import Path
from typing import Iterable, TypeVar

import orjson as json
import ormsgpack

from . import PathLike

T = TypeVar("T")


class DictDb(dict):
    def __init__(self, path: PathLike) -> None:
        super().__init__()
        self.path = Path(path).with_suffix(".msgpack")
        if self.path.with_suffix(".msgpack").exists():
            self.update(ormsgpack.unpackb(self.path.read_bytes()))
        elif (path2 := self.path.with_suffix(".json")).exists():
            self.update(json.loads(path2.read_bytes()))

    def flush(self) -> None:
        self.path.write_bytes(ormsgpack.packb(self))

    def auto_flush(self, iterable: Iterable[T], interval: float = 60) -> Iterable[T]:
        last_flush = time.time()
        flushed = True
        for it in iterable:
            yield it
            t = time.time()
            if t - last_flush > interval:
                last_flush = t
                self.flush()
                flushed = True
            else:
                flushed = False
        if not flushed:
            self.flush()
