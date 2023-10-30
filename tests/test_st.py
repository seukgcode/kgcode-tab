from rapidfuzz import fuzz

from src.analysis import annotate_all
from src.datasets import ShortTablesDataset
from src.evaluators import CEA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process import filters as F
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = ShortTablesDataset("E:/Project/kg-tab/.datasets/Wikidata_GS_ST", limit=10)

    tp = TableProcessor(".cache/st.json")
    EntityManager.load(".cache/st-entities.msgpack")
    tp.add_dataset(ds, force=False)
    # tp.process(
    #     BingRequester(),
    #     WDSearch(),
    #     skip_query=True,
    #     final_filters=[
    #         F.score_by_ratio(fuzz.ratio),
    #         F.order_by(key=lambda c: -c.score / (1 + c.rank)**0.25),
    #     ]
    # )

    # cutoff 没效果
    ans = Answerer(ds)
    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata

    Matcher.alpha = 0.95
    # Matcher.beta = 0.9
    ValueMatchWikidata.match_cutoff = 0.5
    ValueMatchWikidata.property_select_threshold = 0.2
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.25
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0
    TableAnnotator.type_ancestors = True
    ValueMatchWikidata.property_select_threshold = 0.15  # 0.1, 0.15, 0.3 一样

    # TableAnnotator.ancestor_level = al
    ans.answer(
        annotate_all(tp, 15),
        ".result",
        [CEA_Evaluator(ds, True)],
        metadata={
            "new": True,
            "cta cutoff": False,
            "primary": "orig",
            "skip eq": False,
            "cta norm before": "auto",
            "ratio": "ratio",
            "cutoff": 2,
        },
    )
    # EntityManager.dump(".cache/st-entities.msgpack")
    generate_report(".result")


if __name__ == "__main__":
    # with Profile():
    main()
