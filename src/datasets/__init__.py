from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, TypeAlias

import pandas as pd
import polars as pl

from ..utils import PathLike

AnnotationTask: TypeAlias = Literal["cta", "cea", "cpa"]


@dataclass
class TaskDesc:
    task: AnnotationTask
    location: tuple[int, int, int, int]  # 表、列、行、答案位置
    offset: tuple[int, int]
    gt: pl.DataFrame
    target: pl.DataFrame


class TableDataset:
    tables_path: Path
    cta: TaskDesc
    cea: TaskDesc
    cpa: TaskDesc

    def __init__(self, path: PathLike, *, limit: int = 0) -> None:
        self.path = Path(path).absolute()
        self.limit = limit

    def get_desc(self, task: AnnotationTask) -> TaskDesc:
        return getattr(self, task)

    @abstractmethod
    def read_table(self, table_path: Path) -> pd.DataFrame:
        ...

    def iter_tables(self) -> Iterable[tuple[Path, pd.DataFrame]]:
        n = 0
        for p in self.tables_path.iterdir():
            if p.is_file():
                n += 1
                yield p, self.read_table(p)
                if n == self.limit:
                    break


from .imdb import ImdbDataset
from .limaye import LimayeDataset
from .musicbrainz import MusicBrainzDataset
from .semtab import SemTabDataset
from .t2d import T2DDataset