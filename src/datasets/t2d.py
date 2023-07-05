from pathlib import Path

import pandas as pd

from ..utils import PathLike
from . import TaskDesc
from ._other import OtherDataset


class T2DDataset(OtherDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.tables_path = self.path / "tables"
        cea_gt = self.read_gt("cea", lambda p: p.stem.removesuffix(".csv.csv"))
        cpa_gt = self.read_gt("cpa", lambda p: p.stem.removesuffix(".csv.csv"))
        self.cea = TaskDesc("cea", (0, 2, 1, 4), (0, 0), cea_gt, cea_gt)
        self.cpa = TaskDesc("cpa", (0, 2, 1, 4), (0, 0), cpa_gt, cpa_gt)

    def read_table(self, table_path: Path) -> pd.DataFrame:
        # return pl.read_csv(table_path, has_header=True).to_pandas(use_pyarrow_extension_array=False)
        # return pd.DataFrame(pl.read_csv(table_path))
        return pd.read_csv(table_path)