from rapidfuzz import fuzz

from src.analysis import annotate_all
from src.analysis.wikidata.filters import default_select
from src.datasets.limaye import LimayeDataset
from src.evaluators import CEA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process import filters as F
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = LimayeDataset("../kg-tab/.datasets/Limaye_v4", limit=0)

    tp = TableProcessor(".cache/limaye_v4.msgpack")
    EntityManager.load(".cache/limaye-v4-entities.msgpack")
    # tp.add_dataset(ds, force=True)
    tp.add_dataset(ds)

    # tp.process(
    #     BingRequester(),
    #     WDSearch(concurrency=50),
    #     skip_query=False,
    #     # force_correct=True,
    #     force_retrieve=True,
    #     retrieval_filters=[
    #         F.score_by_ratio(fuzz.ratio),
    #         F.order_by(key=lambda c: -c.score / (1 + c.rank)**0.25),
    #         F.limiter(20)
    #     ],
    #     final_filters=[],
    # )

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata

    Matcher.alpha = 0.80
    Matcher.beta = 0.95
    ValueMatchWikidata.match_cutoff = 0.5
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.05
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0.5
    TableAnnotator.type_ancestors = True
    TableAnnotator.ancestor_rate = 0.7
    TableAnnotator.ancestor_level = 2
    ValueMatchWikidata.property_select_threshold = 0.65
    TableAnnotator.multi_subcol = False

    ans = Answerer(ds)

    aa = annotate_all(tp, 20, cea_selector=default_select(primary_eps=1e-2, tertiary_eps=1e-1))
    ans.answer(
        aa,
        ".result",
        [CEA_Evaluator(ds, True)],
        dataset_name="Limaye_v4",
        metadata={
            "sync": "st",
            "props": "late rank",
            "primary eps": 0.01,
            "fix spell": True,
            "tertiary_eps": 1e-1,
            "num sim": "new",
        },
    )
    # EntityManager.dump(".cache/limaye-v4-entities.msgpack")
    generate_report(".result")
    # 这个数据集要用ratio而非partial


if __name__ == "__main__":
    # with Profile():
    main()
