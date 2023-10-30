from pathlib import Path

import pandas as pd

from ..utils import PathLike
from . import TaskDesc
from ._other import OtherDataset


class MusicBrainzDataset(OtherDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.tables_path = self.path / "tables"
        cea_gt = self.read_gt("ground_truth")
        self.cea = TaskDesc("cea", (0, 2, 1, 4), (0, 0), cea_gt, cea_gt)

    def read_table(self, table_path: Path) -> pd.DataFrame:
        # return pd.DataFrame(pl.read_csv(table_path))
        return pd.read_csv(table_path)
