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
from src.datasets import LimayeDataset, ShortTablesDataset, T2DDataset, ToughTablesDataset
from src.process import TableProcessor
from src.process.wikibase import EntityManager
from src.utils.jsonio import read_json, write_json


def entity_ratio(qid: str, s: str, ratio=fuzz.ratio):
    return max((float(ratio(a, s, processor=default_process)) for a in EntityManager.get(qid).iter_aliases()),
               default=.0)


def entity_ratio_many(gts: list[str], s: str, ratios):
    return [max((entity_ratio(gt, s, ratio) for gt in gts), default=0.) for ratio in ratios]


def entity_distance(qid: str, s: str):
    return min((Levenshtein.distance(a, s, processor=default_process) for a in EntityManager.get(qid).iter_aliases()),
               default=0)


def entity_distance_many(gts: list[str], s: str):
    return min((entity_distance(gt, s) for gt in gts), default=0)


def main():
    # ds = ToughTablesDataset("../kg-tab/.datasets/2T_WD")
    # ds = T2DDataset("../kg-tab/.datasets/T2D2")
    ds = LimayeDataset("../kg-tab/.datasets/Limaye_v4")
    # ds = LimayeDataset("../kg-tab/.datasets/Limaye_v3")
    # ds = ShortTablesDataset("../kg-tab/.datasets/Wikidata_GS_ST")

    # tp = TableProcessor(".cache/st.json")
    # tp = TableProcessor(".cache/t2d.json")
    tp = TableProcessor(".cache/limaye_v4.msgpack")
    # tp = TableProcessor(".cache/2t.msgpack")
    tp.add_dataset(ds)

    desc = ds.cea

    at, ac, ar, aa = desc.location

    # all_gt: list[str] = []
    # for row in ds.cea.gt.iter_rows():
    #     all_gt.extend([s[s.rindex('/') + 1 :] for s in str(row[aa]).split() if s and s != "None" and s != "NIL"])
    # EntityManager.store_wikidata_entities(all_gt, False)

    res: list[list[float]] = []
    res2: list[int] = []

    for row in tqdm(ds.cea.gt.iter_rows(), total=len(ds.cea.gt), disable=True):
        tid = str(row[at])
        # if not tid.lower().startswith("00e2h310"): continue
        c, r = row[ac] + desc.offset[0], row[ar] + desc.offset[1]
        gts = [s[s.rindex('/') + 1 :] for s in str(row[aa]).split() if s and s != "None" and s != "NIL"]
        tab = tp.get_table_in_task(tid)

        s = tab[c, r].text
        res.append(entity_ratio_many(gts, s, (fuzz.QRatio, fuzz.partial_ratio, fuzz.WRatio)))
        res2.append(entity_distance_many(gts, s))
        if (res2[-1] > 20):
            print(tid, res2[-1], s, [list(EntityManager.get(qid).iter_aliases()) for qid in gts])

    a = np.array(res)
    b = np.array(res2)
    c = b[b > 0]
    print(c.shape, len(c) / len(b))
    print(a.shape, np.average(a, axis=0), np.std(a, axis=0))
    print(b.shape, np.average(b), np.std(b), np.max(b))
    print(c.shape, np.average(c), np.std(c), np.max(c))
    print(np.bincount(b))
    sns.histplot(pd.DataFrame(c.reshape(-1, 1), columns=["distance"]), x="distance")
    plt.show()


main()