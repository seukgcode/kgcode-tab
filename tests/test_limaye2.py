import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.analysis import annotate_all
from src.datasets.limaye import LimayeDataset
from src.evaluators import CEA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = LimayeDataset(r"E:\Project\kg-tab\.datasets\Limaye_v2", limit=0)

    tp = TableProcessor(".cache/limaye_v2.msgpack")
    EntityManager.load(".cache/limaye-v2-entities.msgpack")
    EntityManager.load(".cache/limaye-entities.msgpack")
    tp.add_dataset(ds, force=False)

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.95
    Matcher.beta = 0.98
    ValueMatchWikidata.match_cutoff = 0.7
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.08
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0.5
    TableAnnotator.type_ancestors = True
    TableAnnotator.ancestor_rate = 0.8
    TableAnnotator.ancestor_level = 4

    ans = Answerer(ds)
    aa = annotate_all(tp, 1)
    EntityManager.dump(".cache/limaye-v2-entities.msgpack")
    ans.answer(aa, ".result", [CEA_Evaluator(ds, True)], metadata={"method": "old"})

    generate_report(".result")


if __name__ == "__main__":
    # with Profile():
    main()
