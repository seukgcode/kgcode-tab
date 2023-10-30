from abc import abstractmethod
from typing import Any, Iterable, Type, overload

import pandas as pd
import polars as pl

from ..datasets import TableDataset, AnnotationTask
from ..datasets.semtab import SemTabDataset
from ..utils import PathLike
from .evaluator import EvalResult, EvaluationException, ensure_startswith


class SemTabEvaluator:
    def __init__(self, task: AnnotationTask, dataset: TableDataset, gt: pl.DataFrame, *, drop_nil: bool) -> None:
        self.task: AnnotationTask = task
        self.dataset = dataset
        self.gt = gt.drop_nulls(gt.columns[-1])
        self.columns = gt.columns
        self.gt_ans = {self._row_key(row): str(row[-1] or "NIL") for row in gt.iter_rows() if not drop_nil or row[-1]}
        self.scope = set[str]()

    def _row_key(self, row: tuple[Any, ...]) -> str:
        return ' '.join(map(str, row[:-1]))

    @abstractmethod
    def evaluate(self, submission_file_path: PathLike) -> EvalResult:
        ...

    def iter_submission(self, submission_file_path: PathLike) -> Iterable[tuple[tuple[Any, ...], str]]:
        self.annotated = set[str]()
        self._err: list[list[str]] = []

        try:
            sub = pl.read_csv(submission_file_path, has_header=False, dtypes=self.gt.dtypes)
        except Exception:
            return

        for row in sub.iter_rows():
            key = self._row_key(row)
            if key not in self.gt_ans:
                continue

            if key in self.annotated:
                print(key)
                raise EvaluationException("Duplicate cells in the submission file")
            else:
                self.annotated.add(key)

            yield row, key

    def append_error(self, row: tuple[Any, ...], col: str) -> None:
        self._err.append([*row[:-1], self.gt_ans[col], row[-1]])

    def append_misses(self) -> None:
        misses = []
        for row in self.gt.iter_rows():
            key = self._row_key(row)
            if key not in self.annotated and key.split()[0] in self.scope:
                misses.append(row)
        df1 = pd.DataFrame(self._err, columns=[*self.columns[:-1], 'gt', self.columns[-1]])
        df2 = pd.DataFrame(misses, columns=[*self.columns[:-1], 'gt'])
        self.err = pd.concat([df1, df2], axis=0)

    def result(self, score: float) -> EvalResult:
        self.append_misses()
        gt_len = len([row[0] for row in self.gt.iter_rows() if row[0] in self.scope])
        return EvalResult.from_score(score, len(self.annotated), gt_len)


class CEA_Evaluator(SemTabEvaluator):
    def __init__(self, dataset: TableDataset, drop_nil: bool = False) -> None:
        super().__init__("cea", dataset, dataset.cea.gt, drop_nil=drop_nil)

    def evaluate(self, submission_file_path: PathLike) -> EvalResult:
        correct_cells = set[str]()

        for row, cell in self.iter_submission(submission_file_path):
            annotation: str = ensure_startswith(row[-1], 'http://www.wikidata.org/entity/')
            ans = self.gt_ans[cell].lower()
            if not ans or ans == "none":
                continue
            if annotation.lower() in ans.split():
                correct_cells.add(cell)
            else:
                self.append_error(row, cell)

        return self.result(len(correct_cells))


class CPA_Evaluator(SemTabEvaluator):
    def __init__(self, dataset: TableDataset, drop_nil: bool = False) -> None:
        super().__init__("cpa", dataset, dataset.cpa.gt, drop_nil=drop_nil)

    def evaluate(self, submission_file_path: PathLike) -> EvalResult:
        correct_cols = set[str]()

        for row, cols in self.iter_submission(submission_file_path):
            annotation: str = ensure_startswith(row[-1], 'http://www.wikidata.org/prop/direct/')
            if annotation.lower() in self.gt_ans[cols].lower().split():
                correct_cols.add(cols)
            else:
                self.append_error(row, cols)

        return self.result(len(correct_cols))


class CTA_Evaluator(SemTabEvaluator):
    gt_ancestor: dict[str, dict[str, str]]
    gt_descendent: dict[str, dict[str, str]]

    def __init__(self, dataset: SemTabDataset, drop_nil: bool = False) -> None:
        super().__init__("cta", dataset, dataset.cta.gt, drop_nil=drop_nil)
        self.gt_ancestor = dataset.gt_ancestor
        self.gt_descendent = dataset.gt_descendent

    def evaluate(self, submission_file_path: PathLike) -> EvalResult:
        total_score = 0

        for row, col in self.iter_submission(submission_file_path):
            annotation: str = ensure_startswith(row[-1], 'http://www.wikidata.org/entity/')

            max_score = self.calc_score(col, annotation)

            if max_score < 0.7:
                self.append_error(row, col)

            total_score += max_score

        return self.result(total_score)

    def calc_score(self, col: str, annotation: str) -> float:
        max_score = 0.0
        for gt_type in self.gt_ans[col].split():
            ancestor = self.gt_ancestor[gt_type]
            ancestor_keys = [k.lower() for k in ancestor]
            descendent = self.gt_descendent[gt_type]
            descendent_keys = [k.lower() for k in descendent]
            if annotation.lower() == gt_type.lower():
                score = 1.0
            elif annotation.lower() in ancestor_keys:
                depth = int(ancestor[annotation])
                score = pow(0.8, depth) if depth <= 5 else 0
            elif annotation.lower() in descendent_keys:
                depth = int(descendent[annotation])
                score = pow(0.7, depth) if depth <= 3 else 0
            else:
                score = 0
            if score > max_score:
                max_score = score
        return max_score