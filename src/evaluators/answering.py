import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl

from ..analysis import AnnotationResult
from ..datasets import TableDataset, TaskDesc, AnnotationTask
from ..evaluators import SemTabEvaluator
from ..utils import PathLike, logger, write_json
from ..utils.mylog import colored


@dataclass
class AnswerSheet:
    task: AnnotationTask
    desc: TaskDesc
    getans: Callable[[AnnotationResult, int, int], str | None]  # 从标注结果获取答案
    evaluator: SemTabEvaluator | None


class Answerer:
    def __init__(self, dataset: TableDataset) -> None:
        self.dataset = dataset

    def _fill_answer(self, desc: TaskDesc, annotations: dict[str, AnnotationResult],
                     drop_nil: bool) -> tuple[pl.DataFrame, set[str]]:
        at, ac, ar, aa = desc.location
        nils = set[str]()
        answers = []

        ans_getter: Callable[[AnnotationResult, int, int], str | None]
        if desc.task == "cta":
            ans_getter = lambda a, c, r: a.cta[c]
        elif desc.task == "cpa":
            ans_getter = lambda a, c, r: a.cpa[r][c]
        else:
            ans_getter = lambda a, c, r: a.cea[c][r]

        ans_sheet = desc.target.unique(maintain_order=True)
        for row in ans_sheet.iter_rows():
            tid = row[at]
            answers.append(list(row))
            answers[-1][aa] = None
            if tid not in annotations:
                nils.add(tid)
                continue
            try:
                ans = ans_getter(annotations[tid], row[ac] + desc.offset[0], row[ar] + desc.offset[1])
                answers[-1][aa] = ans or (None if drop_nil else "NIL")
            except IndexError:
                logger.error("Index Error: %s", row)

        if not answers:
            logger.warning("No answer provided! Please check your configuration.")

        answer_df = pl.DataFrame(answers, schema=ans_sheet.schema)
        answer_df = answer_df.drop_nulls(subset=[answer_df.columns[aa]])

        return answer_df, nils

    def _answer_impl(
        self,
        annotations: dict[str, AnnotationResult],
        dst_path: Path,
        evaluator: SemTabEvaluator,
        *,
        drop_nil: bool = True,
    ) -> dict[str, Any]:
        '''dst_path是输出目录而非文件'''
        task = evaluator.task
        logger.info(f"Answering: {task}")

        df, nils = self._fill_answer(self.dataset.get_desc(task), annotations, drop_nil)

        dst_path.mkdir(parents=True, exist_ok=True)
        out_path = dst_path / f"{task}.csv"
        df.write_csv(out_path, has_header=False)

        if len(nils) > 0:
            logger.warning(f"Tables not found ({len(nils)})")

        res: dict[str, Any] = {"missing": list(nils)}

        eval_res = evaluator.evaluate(out_path)
        logger.info(colored(eval_res, fg=5, bright=True, flash=True))
        res |= eval_res.to_dict()
        evaluator.err.to_csv(dst_path / f"{task}-err.csv", header=False, index=False)

        return res

    def answer(
        self,
        results: tuple[dict[str, AnnotationResult], dict[str, float]],
        dst_path: PathLike,
        evaluators: list[SemTabEvaluator],
        *,
        drop_nil: bool = True,
        metadata: Any = None
    )  -> dict[str, Any]:
        now = datetime.datetime.now()
        dst_path = Path(dst_path).absolute() / "history" / now.strftime("%Y%m%d %H%M%S")
        dst_path.mkdir(parents=True)

        logger.info("Start answering and evaluating...")
        report: dict[str, Any] = {
            "datetime": str(now), "dataset": self.dataset.__class__.__name__.removesuffix("Dataset"), "parameters":
            results[1], "metadata": metadata,
            **{ev.task.upper(): self._answer_impl(results[0], dst_path, ev, drop_nil=drop_nil)
               for ev in evaluators}
        }
        write_json(report, dst_path / "report.json", indent=2)

        try:
            latest = dst_path.parents[1] / "latest"
            latest.unlink(True)
            latest.symlink_to(dst_path, True)
        except:
            ...

        logger.info("Evaluation completed.")
        return report
