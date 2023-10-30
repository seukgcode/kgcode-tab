from rapidfuzz import fuzz

from src.analysis import annotate_all
from src.analysis.wikidata.filters import default_select
from src.datasets import ToughTablesDataset
from src.evaluators import CEA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process import filters as F
from src.process.correctors import BingRequester, WikipediaRequester
from src.process.wikisearch import WDSearch


def main():
    ds = ToughTablesDataset("../kg-tab/.datasets/2T_WD", limit=10)

    tp = TableProcessor(".cache/2t.msgpack")
    EntityManager.load(".cache/2t-entities.msgpack")
    # tp.add_dataset(ds, force=True)
    # tp.add_dataset(ds)
    tp.add_table(r"E:\Project\kg-tab\.datasets\2T_WD\tables\00E2H310.csv")

    # tp.process(
    #     WikipediaRequester(concurrency=150),
    #     WDSearch(concurrency=50),
    #     # skip_query=False,
    #     # force_correct=True,
    #     force_retrieve=True,
    #     retrieval_filters=[
    #         F.score_by_ratio(fuzz.partial_ratio),
    #         F.order_by(key=lambda c: -c.score / (1 + c.rank)**0.25),
    #         F.limiter(10)
    #     ]
    # )

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.8
    Matcher.beta = 0.98
    ValueMatchWikidata.match_cutoff = 0.7
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.05
    TableAnnotator.infer_subcol = False
    ValueMatchWikidata.key_threshold = 0.5
    TableAnnotator.type_ancestors = True
    TableAnnotator.ancestor_rate = 0.8
    TableAnnotator.ancestor_level = 2
    ValueMatchWikidata.property_select_threshold = 0.15
    TableAnnotator.multi_subcol = False

    ans = Answerer(ds)

    aa = annotate_all(tp, 10, cea_selector=default_select(primary_eps=1e-2, tertiary_eps=1e-1))
    ans.answer(
        aa,
        ".result",
        [CEA_Evaluator(ds, True)],
        dataset_name="ToughTables",
        metadata={},
    )
    EntityManager.dump(".cache/2t-entities.msgpack")
    generate_report(".result")


if __name__ == "__main__":
    # with Profile():
    main()
