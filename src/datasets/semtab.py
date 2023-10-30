import json
from pathlib import Path

import pandas as pd
import polars as pl
from typing_extensions import override

from ..datasets import TableDataset, TaskDesc
from ..utils import PathLike


class SemTabDataset(TableDataset):
    def __init__(self, tables_path: PathLike, target_path: PathLike, gt_path: PathLike, **kwargs) -> None:
        super().__init__(Path(tables_path).parent, **kwargs)
        self.tables_path = Path(tables_path)
        self.target_path = Path(target_path)
        self.gt_path = Path(gt_path)
        cta_gt = self.read_sheet(self.gt_path / "cta_gt.csv")
        cea_gt = self.read_sheet(self.gt_path / "cea_gt.csv")
        cpa_gt = self.read_sheet(self.gt_path / "cpa_gt.csv")
        self.cta = TaskDesc("cta", (0, 1, 1, 2), (0, -1), cta_gt, cta_gt)
        self.cpa = TaskDesc("cpa", (0, 2, 1, 3), (0, 0), cpa_gt, cpa_gt)
        self.cea = TaskDesc("cea", (0, 2, 1, 3), (0, -1), cea_gt, cea_gt)
        self.gt_ancestor = json.loads((self.gt_path / "cta_gt_ancestor.json").read_text())
        self.gt_descendent = json.loads((self.gt_path / "cta_gt_descendent.json").read_text())

    def read_sheet(self, path: PathLike) -> pl.DataFrame:
        return pl.read_csv(path, has_header=False)

    @override
    def read_table(self, table_path: Path) -> pd.DataFrame:
        return pd.read_csv(table_path)
