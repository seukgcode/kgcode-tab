from itertools import product
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
    EntityManager.load(".cache/t2d-entities.msgpack")
    tp.add_dataset(ds)
    # tp.process(BingRequester(), WDSearch())

    # cutoff 没效果
    ans = Answerer(ds)
    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.95
    Matcher.beta = 0.98
    ValueMatchWikidata.match_cutoff = 0.7
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.005
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0
    TableAnnotator.type_ancestors = True

    for t in (True, ):
        TableAnnotator.infer_subcol = t
        ans.answer(
            annotate_all(tp, 3),
            ".result", [CEA_Evaluator(ds, True), CPA_Evaluator(ds, True)],
            metadata={
                "method": "old", "number match": True, "aliases match": True, "prop rating": True, "match time": True,
                "fix coord": True
            }
        )
    # EntityManager.dump(".cache/t2d-entities.msgpack")
    generate_report(".result")
    # cpa分数高需要valuematch对数值做处理


if __name__ == "__main__":
    # with Profile():
    main()
