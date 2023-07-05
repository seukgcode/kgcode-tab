import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.analysis import annotate_all
from src.datasets.semtab import SemTabDataset
from src.evaluators import CEA_Evaluator, CPA_Evaluator, CTA_Evaluator
from src.evaluators.report import generate_report
from src.evaluators.answering import Answerer
from src.process import EntityManager, TableProcessor
from src.utils import Profile
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch


def main():
    ds = SemTabDataset(
        r"E:\Project\kg-tab\data\HT-Valid-mixed\DataSets\Valid\tables",
        r"E:\Project\kg-tab\data\HT-Valid-mixed\DataSets\Valid\gt",
        r"E:\Project\kg-tab\data\HT-Valid-mixed\DataSets\Valid\gt",
        limit=0
    )

    tp = TableProcessor(".cache/sample-tables.json")
    EntityManager.load(".cache/ht-entities.msgpack")

    tp.add_dataset(ds)
    # tp.add_table(r"E:\Project\kg-tab\.datasets\1.csv")
    tp.process(BingRequester(concurrency=100), WDSearch(concurrency=50, chunk_size=100), force_retrieve=True)

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.95
    Matcher.beta = 0.98
    ValueMatchWikidata.match_cutoff = 0.8
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.01
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0.5
    TableAnnotator.ancestor_level = 3
    TableAnnotator.type_ancestors = True

    ans = Answerer(ds)
    ans.answer(
        annotate_all(tp, 100),
        ".result", [CTA_Evaluator(ds), CEA_Evaluator(ds), CPA_Evaluator(ds)],
        metadata={"method": "old"}
    )
    EntityManager.dump(".cache/ht-entities.msgpack")

    generate_report(".result")


if __name__ == "__main__":
    # with Profile():
    main()
