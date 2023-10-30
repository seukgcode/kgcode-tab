from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import PathLike
from . import TaskDesc
from ._other import OtherDataset


class LimayeDataset(OtherDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.tables_path = self.path / "tables"
        cea_gt = self.read_gt("ground_truth", lambda p: p.stem.removesuffix(".cell.keys"))
        self.cea = TaskDesc("cea", (0, 2, 1, 4), (0, 0), cea_gt, cea_gt)

    def read_table(self, table_path: Path) -> pd.DataFrame:
        # return pl.read_csv(table_path, has_header=True).to_pandas(use_pyarrow_extension_array=False)
        # return pd.DataFrame(pl.read_csv(table_path))
        df = pd.read_csv(table_path, header=None if table_path.stem.startswith("file") else "infer", na_values=["N/A"])
        return df.applymap(LimayeDataset._parse_url)

    @staticmethod
    def _parse_url(s: Any):
        if isinstance(s, str):
            return " ".join(filter(lambda a: not a.startswith("../"), s.split()))
        return s
