from itertools import product
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.analysis import annotate_all
from src.datasets.t2d import T2DDataset
from src.evaluators import CEA_Evaluator, CPA_Evaluator
from src.evaluators.answering import Answerer
from src.evaluators.report import generate_report
from src.process import EntityManager, TableProcessor
from src.process.correctors import BingRequester
from src.process.wikisearch import WDSearch
from src.utils import Profile


def main():
    ds = T2DDataset(r"E:\Project\kg-tab\.datasets\T2D2", limit=0)

    tp = TableProcessor(".cache/t2d.json")
    # EntityManager.load(".cache/t2d-entities.msgpack")
    tp.add_dataset(ds, force=False)
    # tp.process(BingRequester(), WDSearch(), candidate_limit=10)

    # tab = tp.get_table("10630177_0_4831842476649004753")
    # tab = tp.get_table("11278409_0_3742771475298785475")
    tab = tp.get_table("1146722_1_7558140036342906956")

    from src.analysis.wikidata.annotator import TableAnnotator
    from src.analysis.wikidata.valuematch import Matcher, ValueMatchWikidata
    Matcher.alpha = 0.95
    Matcher.beta = 0.9
    ValueMatchWikidata.match_cutoff = 0.7
    TableAnnotator.normal_factor = 1
    TableAnnotator.fallback_threshold = 0.01
    TableAnnotator.infer_subcol = True
    ValueMatchWikidata.key_threshold = 0
    TableAnnotator.type_ancestors = True
    a = TableAnnotator(tab).process()
    return


if __name__ == "__main__":
    # with Profile():
    main()
