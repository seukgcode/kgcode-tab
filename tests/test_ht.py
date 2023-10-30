from rapidfuzz import fuzz

from src.analysis import annotate_all
from src.datasets.semtab import SemTabDataset
from src.evaluators import CEA_Evaluator, CPA_Evaluator, CTA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process import filters as F
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


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
    tp.add_table(r"E:\Project\kg-tab\.datasets\1.csv", force=True)
    tp.process(
        BingRequester(),
        WDSearch(concurrency=50),
        skip_query=False,
        # force_correct=True,
        force_retrieve=True,
        retrieval_filters=[
            F.score_by_ratio(fuzz.ratio),
            F.order_by(key=lambda c: -c.score / (1 + c.rank)**0.25),
            F.limiter(100)
        ],
        final_filters=[],
    )

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.9
    Matcher.beta = 0.95
    ValueMatchWikidata.match_cutoff = 0.6
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.01
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0.5
    TableAnnotator.ancestor_level = 5
    TableAnnotator.type_ancestors = True
    ValueMatchWikidata.property_select_threshold = 0.2

    ans = Answerer(ds)

    for ft in (0.05,):
        # ValueMatchWikidata.property_select_threshold = ft

        ans.answer(
            annotate_all(tp, 50), ".result",
            [CTA_Evaluator(ds), CEA_Evaluator(ds), CPA_Evaluator(ds)], metadata={}
        )
    # EntityManager.dump(".cache/ht-entities.msgpack")

    generate_report(".result")


if __name__ == "__main__":
    # with Profile():
    main()
