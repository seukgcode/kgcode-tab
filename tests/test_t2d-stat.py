import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.analysis import annotate_all, annotate_all_mt
from src.datasets.t2d import T2DDataset
from src.evaluators import CEA_Evaluator, CPA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = T2DDataset(r"E:\Project\kg-tab\.datasets\T2D2")

    tp = TableProcessor(".cache/t2d.json")
    tp.add_dataset(ds)
    cell_cnt = 0
    cand_cnt = 0
    cand_cnt_max = 0
    for t in tp.tasks:
        for c in t.table:
            cell_cnt += sum(bool(s) for s in c.cell_texts)
            for ce in c:
                cand_cnt += len(ce.candidates)
                cand_cnt_max = max(cand_cnt_max, len(ce.candidates))
    print(cell_cnt, cand_cnt, cand_cnt / cell_cnt, cand_cnt_max)


if __name__ == "__main__":
    # with Profile():
    main()
