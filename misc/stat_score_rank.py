import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.datasets import LimayeDataset, ShortTablesDataset, T2DDataset, TableDataset, ToughTablesDataset
# from src.process import TableProcessor
# from src.process.wikibase import EntityManager
from src.utils.jsonio import read_json, write_json

np.set_printoptions(linewidth=160)


def trans_gt(ds: TableDataset):
    desc = ds.cea

    at, ac, ar, aa = desc.location

    res = dict[tuple[str, int, int], set[str]]()

    for row in ds.cea.gt.iter_rows():
        tid = str(row[at])
        # if not tid.lower().startswith("00e2h310"): continue
        c, r = row[ac] + desc.offset[0], row[ar] + desc.offset[1]
        gts = {s[s.rindex('/') + 1 :] for s in str(row[aa]).split() if s and s != "None" and s != "NIL"}
        res[(tid, c, r)] = gts

    return res


def combine(a: list[str], b: list[str], k: int):
    c = a[:]
    for x in b:
        if x not in c:
            c.append(x)
    return c


def main():
    # ds = T2DDataset("../kg-tab/.datasets/T2D2")
    # fs = pd.read_csv(".result/history/20230816 152248/full_score.csv")
    # ranking = read_json(".result/history/20230816 152248/cea_ranking.json")

    # ds = LimayeDataset("../kg-tab/.datasets/Limaye_v4")
    # fs = pd.read_csv(".result/history/20230815 233509/full_score.csv")
    # ranking = read_json(".result/history/20230816 145731/cea_ranking.json")

    ds = ShortTablesDataset("../kg-tab/.datasets/Wikidata_GS_ST")
    fs = pd.read_csv(".result/history/20230816 154154/full_score.csv")
    ranking = read_json(".result/history/20230816 154154/cea_ranking.json")

    gt = trans_gt(ds)

    fs["table"] = fs["table"].astype(str)
    fs["natural_rank"] = -fs["natural_rank"]

    # print(fs)
    crits = [
        "primary", "secondary", "tertiary_0", "tertiary_1", "natural_rank", "match", "textual_Q", "textual_W",
        "textual_partial", "textual_token"
    ]  # 除了rank其他都是越高越好
    # Limaye 是 textual_token 最好

    tot = 0
    # top1: list[np.ndarray] = []
    tops = np.zeros((len(crits) + 1, 10))
    for ((tab, r, c), data) in fs.groupby(["table", "row", "col"]):
        # print(tab, r, c)
        this_gt = gt[(tab, c, r)]
        tot += 1
        for i, crit in enumerate(crits):
            l = data.sort_values(crit, axis=0, ascending=False)
            rank = combine(l["qid"].tolist(), ranking[tab][c][r], len(l))
            # top[i] = bool(rank[])
            for k in range(10):
                tops[i, k] += bool(this_gt & set(rank[: k + 1]))
        rank = ranking[tab][c][r]
        for k in range(10):
            tops[len(crits), k] += bool(this_gt & set(rank[: k + 1]))

    print(tops)
    print(tops / tot)

    np.set_printoptions(linewidth=160, precision=4)

    print((tops / tot)[[10, 5, 0, 1, 2, 9, 4]][:, [0, 4, 9]])


main()