import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import TypeVar

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import LimayeDataset, ShortTablesDataset, T2DDataset
from src.utils.jsonio import read_json, write_json

T = TypeVar('T')


def trans(a: dict[str, list[list[T]]]) -> list[list[dict[str, T]]]:
    first = next(iter(a.items()))
    keys = a.keys()
    l1 = len(first[1])
    l2 = len(first[1][0])
    return [[{k: a[k][i][j] for k in keys} for j in range(l2)] for i in range(l1)]


def swap_inout(a: dict[str, dict[str, T]]) -> defaultdict[str, dict[str, T]]:
    res = defaultdict[str, dict[str, T]](dict)
    for key, val in a.items():
        for key_in, val_in in val.items():
            res[key_in][key] = val_in
    return res


def main():
    # ds = T2DDataset(".datasets/T2D2")
    # ds = T2DDataset("../kg-tab/.datasets/T2D2")
    # ds = LimayeDataset("../kg-tab/.datasets/Limaye_v4")
    # ds = LimayeDataset("../kg-tab/.datasets/Limaye_v3")
    ds = ShortTablesDataset("../kg-tab/.datasets/Wikidata_GS_ST")

    # scores = read_json(".result/latest/cea_score.json")
    result_path = Path(".result/history/20230816 154154")
    # result_path = Path(".result/latest/")
    scores = read_json(result_path / "cea_score.json")

    data = []

    for row in tqdm(ds.cea.gt.iter_rows(), total=len(ds.cea.gt)):
        gt_str = row[ds.cea.location[3]]
        if not gt_str or str(gt_str) == "nan":
            continue
        # if str(row[4]).count('Q') > 1:
        #     print(row[4])
        gts = [s[s.rindex('/') + 1 :] for s in str(gt_str).split()]
        if not gts:
            continue
        r, c = row[1], row[2]
        tr = trans(scores[str(row[0])])
        trs = swap_inout(tr[c][r])

        data.extend({"table": row[0], "row": r, "col": c, "qid": k, **v, "label": k in gts} for k, v in trs.items())

    df = pd.DataFrame(data)
    df.to_csv(result_path / "full_score.csv", index=False)

main()