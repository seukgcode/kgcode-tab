from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

from ..utils import PathLike, logger
from . import TableDataset


class OtherDataset(TableDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)

    def read_gt(self, folder: str, gt_name_processor: Callable[[Path], str] = lambda p: p.stem) -> pl.DataFrame:
        cached = (self.path / folder).with_suffix(".parquet")
        if cached.exists():
            return pl.read_parquet(cached)
        logger.info("Make dataset gt parquet: %s", cached)
        gts = []
        for p in (self.path / folder).iterdir():
            if p.stat().st_size <= 0:
                continue
            df = pl.read_csv(p, has_header=False)
            df.insert_at_idx(0, pl.Series("table", [gt_name_processor(p)] * len(df)))
            gts.append(df)
        gt = pl.concat(gts)
        gt.write_parquet(cached)
        # gt.write_csv(cached.with_suffix(".csv")) # csv is too large!
        return gt

    def read_table(self, table_path: Path) -> pd.DataFrame:
        # return pd.DataFrame(pl.read_csv(table_path))
        return pd.read_csv(table_path, header=None)


class CEAOnlyDataset(OtherDataset):
    def __init__(self, path: PathLike, **kwargs) -> None:
        super().__init__(path, **kwargs)
