import math
import time

from tqdm import tqdm

from ..process import TableProcessor
from ..utils import logger
from .annotation_result import AnnotationResult
from .wikidata.annotator import TableAnnotator
from .wikidata.valuematch import Matcher, ValueMatchWikidata


def annotate_all_mt(proc: TableProcessor, max_workers: int = 8) -> dict[str, AnnotationResult]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger.info("Start annotating tables.")
    results: dict[str, AnnotationResult] = {}
    # with ThreadPoolExecutor(max_workers) as executor:
    #     tasks = [executor.submit(TableAnnotator(t.table).process) for t in proc.tasks]
    #     for t in tqdm(as_completed(tasks), total=len(tasks), colour="#7B68EE"):
    #         ta = t.result()
    #         results[ta.table.name] = AnnotationResult((ta.table.rows, ta.table.cols),
    #                                                   ta.table.key_col,
    #                                                   ta.cta,
    #                                                   ta.cea,
    #                                                   ta.cpa,
    #                                                   ta.all_cea_scores,
    #                                                   duration=0)
    return results


def annotate_all(proc: TableProcessor,
                 candidates_limit: int = -1,
                 **kwargs) -> tuple[dict[str, AnnotationResult], dict[str, float]]:
    logger.info("Start annotating tables.")
    annotators: dict[str, AnnotationResult] = {}
    xx = []
    t1 = time.time()
    fb = 0
    for task in tqdm(proc.tasks, colour="#1E90FF"):
        task.limit_candidates(candidates_limit)
        table = task.table
        t10 = time.time_ns()
        ta = TableAnnotator(table).process(**kwargs)
        t11 = time.time_ns()
        if table.key_col != 0:
            xx.append((table.name, [(col.metadata["major"] if col.metadata else -1) for col in table]))
        result = AnnotationResult(
            shape=(table.rows, table.cols),
            subcol=table.key_col,
            cta=ta.cta,
            cea=ta.cea,
            cpa=ta.cpa,
            cea_score=ta.all_cea_scores, # type: ignore
            cea_ranking=ta.cea_ranking,
            duration=t11 - t10
        )
        annotators[table.name] = result
        if all(ta.cta_fallback):
            fb += 1
    t2 = time.time()
    params = {
        "candidates_limit": candidates_limit, **Matcher.params, **ValueMatchWikidata.params, **TableAnnotator.params,
        "duration": math.floor((t2 - t1) * 1000) / 1000, "all_fallback": fb
    }
    return annotators, params
