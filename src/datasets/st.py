from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from ..utils import PathLike, logger, read_json
from . import TableDataset, TaskDesc


class ShortTablesDataset(TableDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.tables_path = self.path / "Tables"
        cea_gt = self.read_gt("GS")
        self.cea = TaskDesc("cea", (0, 2, 1, 6), (0, 0), cea_gt, cea_gt)

    def read_gt(self, folder: str) -> pl.DataFrame:
        cached = (self.path / folder).with_suffix(".parquet")
        if cached.exists():
            return pl.read_parquet(cached)
        logger.info("Make dataset gt parquet: %s", cached)
        gts = []
        for p in (self.path / folder).iterdir():
            df = pl.read_csv(p, has_header=False)
            tab = self.read_raw_table((self.tables_path / p.name).with_suffix(".json"))
            kci = tab["keyColumnIndex"]
            df.drop_in_place("column_5")
            df = df.apply(lambda t: (*t[:-1], "http://www.wikidata.org/entity/" + t[-1]))
            df.insert_at_idx(0, pl.Series("table", [int(p.stem)] * len(df)))
            df.insert_at_idx(1, pl.Series("row", list(range(len(df)))))
            df.insert_at_idx(2, pl.Series("col", [kci] * len(df)))
            gts.append(df)
        gt = pl.concat(gts)
        gt = gt.sort(["table", "row", "col"])
        gt = gt.replace("table", gt["table"].apply(str))
        gt.write_parquet(cached)
        gt.write_csv(cached.with_suffix(".csv"))  # csv is too large!
        return gt

    def read_table(self, table_path: Path) -> pd.DataFrame:
        rel = self.read_raw_table(table_path)["relation"]
        return pd.DataFrame(rel, columns=[f"column_{i}" for i in range(len(rel[0]))])

    def read_raw_table(self, table_path: Path) -> Any:
        return read_json(table_path)
