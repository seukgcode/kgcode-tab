from rapidfuzz import fuzz

from src.analysis import annotate_all
from src.analysis.wikidata import filters as Fa
from src.datasets.t2d import T2DDataset
from src.evaluators import CEA_Evaluator, CPA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process import filters as F
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = T2DDataset("E:/Project/kg-tab/.datasets/T2D2", limit=0)

    tp = TableProcessor(".cache/t2d.json")
    EntityManager.load(".cache/t2d-entities.msgpack")
    tp.add_dataset(ds, force=False)
    tp.process(
        BingRequester(),
        WDSearch(concurrency=50),
        skip_query=False,
        # force_correct=True,
        force_retrieve=True,
        retrieval_filters=[
            F.score_by_ratio(fuzz.ratio),
            F.order_by(key=lambda c: -c.score / (1 + c.rank) ** 0.25),
            F.limiter(15),
        ],
        final_filters=[],
    )

    # cutoff 没效果
    ans = Answerer(ds)
    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata

    Matcher.alpha = 0.9
    # Matcher.beta = 0.9
    ValueMatchWikidata.match_cutoff = 0.7
    ValueMatchWikidata.property_select_threshold = 0.15
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.25
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0
    TableAnnotator.type_ancestors = True

    for mc in (0.7,):
        ValueMatchWikidata.match_cutoff = mc
        ans.answer(
            annotate_all(tp, 15, cea_selector=Fa.default_select(eq_skip=False)),
            ".result",
            [CEA_Evaluator(ds, True), CPA_Evaluator(ds, True)],
            metadata={"num sim": "new"},
        )
    # EntityManager.dump(".cache/t2d-entities.msgpack")
    generate_report(".result")
    # cpa分数高需要valuematch对数值做处理
    # 该数据集不需要重新处理cutoff


if __name__ == "__main__":
    # with Profile():
    main()
