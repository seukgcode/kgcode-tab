from pathlib import Path

import pandas as pd
import polars as pl
from typing_extensions import override

from ..datasets import TableDataset, TaskDesc
from ..utils import PathLike


class ToughTablesDataset(TableDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.tables_path = Path(path) / "tables"
        cea_gt = self.read_sheet(Path(path) / "gt" / "CEA_2T_WD_gt.csv")
        self.cea = TaskDesc("cea", (0, 2, 1, 3), (0, -1), cea_gt, cea_gt)

    def read_sheet(self, path: PathLike) -> pl.DataFrame:
        return pl.read_csv(path, has_header=False)

    @override
    def read_table(self, table_path: Path) -> pd.DataFrame:
        return pd.read_csv(table_path)
