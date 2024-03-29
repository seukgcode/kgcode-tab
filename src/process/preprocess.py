import re
import uuid
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from more_itertools import first_true, flatten, prepend
from rapidfuzz import fuzz
from tqdm import tqdm

from ..datasets import TableDataset
from ..table import Candidate, Cell, Column, Table
from ..table.table_data import SpacyType
from ..utils import DictDb, PathLike, config, logger
from ..utils.lists import make_list
from ..utils.strings import despaces
from . import filters as F
from .correctors import CheckRequester, SpellChecker
from .filters import CandidateFilter
from .wikibase import EntityManager
from .wikisearch import SearchHelper

logger.info("Load preprocess module")

LABEL_: dict[SpacyType, int] = {
    "PERSON": 0,
    "NORP": 0,
    "FAC": 0,
    "ORG": 0,
    "GPE": 0,
    "LOC": 0,
    "PRODUCT": 0,
    "EVENT": 0,
    "WORK_OF_ART": 0,
    "LAW": 0,
    "LANGUAGE": 0,
    "DATE": 0,
    "TIME": 0,
    "PERCENT": 0,
    "MONEY": 0,
    "QUANTITY": 0,
    "ORDINAL": 0,
    "CARDINAL": 0,
}

_digits_pat = re.compile(r"[0-9/.+-]+%?")


def _is_numerical(series: pd.Series):
    tot = sum(bool(str(x)) for x in series)
    cnt = sum(bool(_digits_pat.fullmatch(str(x))) for x in series)
    return tot == 0 or cnt / tot > 0.9 or cnt > 2 and cnt >= tot - 1


class TableTask:
    def __init__(self, table: Table) -> None:
        self.table = table

    @classmethod
    def from_dataframe(cls, data_frame: pd.DataFrame, *, path: PathLike | None = None, name: str = ""):
        data_frame_nona = data_frame.fillna("")
        data: list[list[object]] = data_frame_nona.transpose().values.tolist()
        has_header = any(not str(s).startswith("col") for s in data_frame.columns.values)
        table = Table(
            path=str(path) if path else None,
            name=name or str(uuid.uuid1()),
            rows=len(data[0]),
            cols=len(data),
            columns=[
                Column(
                    empty=False,
                    searchable=False,
                    numerical=data_frame.dtypes[i] != object or _is_numerical(data_frame_nona.iloc[:, i]),
                    type=None,
                    name=str(data_frame.columns.values[i]) if has_header else None,
                    cells=[
                        Cell(
                            is_none=cell is None or cell == "",
                            text=str(cell),
                            value=despaces(str(cell)),
                            corrections=[],
                            candidates=[],
                        )
                        for cell in col
                    ],
                )
                for i, col in enumerate(data)
            ],
            row_texts=["; ".join(map(str, filter(bool, r))) for _, r in data_frame_nona.iterrows()],
        )
        return cls(table)

    @classmethod
    def from_csv(cls, path: PathLike, sep: str | None = ","):
        return cls.from_dataframe(pd.read_csv(path, sep=sep), name=Path(path).stem, path=Path(path).absolute())

    def init_process(self, *, force: bool = False) -> bool:
        """_summary_

        Args:
            force (bool, optional): _description_. Defaults to False.

        Returns:
            bool: Whether updated
        """
        if self.table.processed and not force:
            return False
        self.judge_columns_category()
        self.judge_key_column()
        self.table.processed = True
        return True

    def judge_columns_category(self) -> None:
        """Judge the category of all tables columns."""

        # judge all columns type

        def is_of_url(col: Column):
            return any("../" in s for s in col.cell_texts)

        # from .nlp_model import spacy_nlp
        self.table.searchable = False
        num_pat = re.compile(r"[0-9/.-]+%?")
        for col_index, col in enumerate(self.table.columns):
            col.searchable = False
            # label_ = LABEL_.copy()
            # none_ = 0
            # for cell in self.table[col_index]:
            #     if cell is None or cell == "":
            #         none_ += 1
            #         continue
            #     doc = spacy_nlp(cell.value)
            #     for entity in doc.ents:
            #         label_[entity.label_] += 1  # type: ignore

            # NE_type = label_['PERSON'] + label_['NORP'] + label_['FAC'] + label_['ORG'] \
            #           + label_['GPE'] + label_['LOC'] + label_['PRODUCT'] + label_['EVENT'] \
            #           + label_['WORK_OF_ART'] + label_['LAW'] + label_['LANGUAGE']
            # literal_type = label_['DATE'] + label_['TIME'] + label_['PERCENT'] + label_['MONEY'] \
            #                + label_['QUANTITY'] + label_['ORDINAL'] + label_['CARDINAL'] + none_
            # type_ = max(label_, key=lambda k: label_[k])
            # col.type = type_
            if is_of_url(col):
                logger.warn("ohhh! %s", self.table.name)
            if not col.numerical and not is_of_url(col):
                col.searchable = True
                self.table.searchable = True

    def judge_key_column(self) -> None:
        # 这个做不了
        if not self.table.searchable:
            logger.warning("Table not searchable: %s", self.table.name)
            return
        self.table.key_col = next((i for i, col in enumerate(self.table.columns) if col.searchable), 0)

    def retrieve_data_fwd(self, searcher: SearchHelper):
        for col in self.table.columns:
            if not col.searchable:
                for cell in col.cells:
                    cell.corrections = []
                    cell.candidates = []
                continue
            for cell in col.cells:
                if cell.is_none:
                    continue
                cell.corrections = SpellChecker.get(cell.value) + [cell.value.lower()]
                # 这里要存什么？返回
                cc = searcher.get(cell.corrections)
                cell.candidates = [Candidate(qid=c[0], match=c[1], rank=c[2], score=0) for c in cc]
                cell.backref()
                yield from (ca.qid for ca in cell.candidates)

    def retrieve_data(self, searcher: SearchHelper, *, force: bool = False, filters: Sequence[CandidateFilter]) -> None:
        if self.table.retrieved and not force:
            return
        for col in self.table.columns:
            for cell in col.cells:
                if cell.is_none:
                    continue
                # cell.corrections = SpellChecker.get(cell.value) + [cell.value.lower()]
                # cc = searcher.get(cell.corrections)
                # # cc = searcher.get(cell.value)
                # cell.candidates = [Candidate(qid=c[0], match=c[1], rank=c[2], score=0) for c in cc]
                # cell.backref()
                cell.candidates_ini = cell.candidates = F.pipe(filters, cell.candidates)
        self.table.retrieved = True

    def finalize(self, *, force: bool = False, filters: Sequence[CandidateFilter]):
        if self.table.completed and not force:
            return

        for cell in self.table.iter_cells():
            if cell.is_none or not cell.candidates:
                continue
            cell.candidates_ini = cell.candidates = F.pipe(prepend(F.filter_ambiguity, filters), cell.candidates)

        self.table.completed = True

    def limit_candidates(self, limit: int = -1):
        if limit < 0:
            return
        for c in self.table:
            for ce in c:
                ce.candidates = ce.candidates_ini[:limit]


class TableProcessor:
    def __init__(self, db_path: PathLike) -> None:
        self.tasks: list[TableTask] = []
        logger.info("Init table processor.")
        self.db = DictDb(db_path)
        logger.info("Found %d tables in DB.", len(self.db))

    """
    看name，如果已经在db里了，且不是force，用已有的
    """

    def _add_table_from_dataframe(self, df: pd.DataFrame, path: Path | None, force: bool) -> bool:
        if path and not force and path.stem in self.db:
            self.tasks.append(TableTask(Table.from_dict(self.db[path.stem])))
        else:
            self.tasks.append(TableTask.from_dataframe(df, path=path, name=path.stem if path else ""))
        return True

    def _add_table_from_path(self, path: Path, force: bool) -> bool:
        if not path.is_file():
            return False
        if path.stem in self.db and not force:
            self.tasks.append(TableTask(Table.from_dict(self.db[path.stem])))
        else:
            self.tasks.append(TableTask.from_csv(path))
        return True

    def add_table(self, table: pd.DataFrame | PathLike, *, path: Path | None = None, force: bool = False) -> bool:
        if isinstance(table, pd.DataFrame):
            return self._add_table_from_dataframe(table, path=path, force=force)
        else:
            return self._add_table_from_path(Path(table), force=force)

    def add_tables(self, tables: list[pd.DataFrame | PathLike] | PathLike, force: bool = False) -> int:
        # 如果tables是路径，加入目录下的表格
        old_len = len(self.tasks)
        add_num = 0
        logger.info("Adding tables from %s ...", tables)
        if isinstance(tables, list):
            add_num = len(tables)
            for table in tqdm(tables, colour="#FFDAB9"):
                self.add_table(table, force=force)
        elif (path := Path(tables)).exists():
            for f in tqdm(list(path.iterdir()), colour="#FFDAB9"):
                add_num += self.add_table(f, force=force)
        logger.info("Add %s/%s tables.", len(self.tasks) - old_len, add_num)
        return add_num

    def add_dataset(self, dataset: TableDataset, force: bool = False) -> int:
        logger.info("Adding tables from dataset.")
        add_num = 0
        for p, df in dataset.iter_tables():
            self.add_table(df, path=p, force=force)
            add_num += 1
        logger.info("Add %d tables.", add_num)
        return add_num

    def get_table(self, name: str) -> Table:
        return Table.from_dict(self.db[name])

    def get_table_in_task(self, name: str) -> Table:
        res = first_true(self.tasks, pred=lambda t: t.table.name == name)
        assert res
        return res.table

    def save_table(self, table: Table, flush: bool = False) -> None:
        self.db[str(table.name)] = table.to_dict()
        if flush:
            self.db.flush()

    def save_task(self, task: TableTask, flush: bool = False) -> None:
        self.save_table(task.table)

    def flush(self) -> None:
        self.db.flush()

    def process(
        self,
        spell_checkers: list[CheckRequester] | CheckRequester,
        searchers: list[SearchHelper] | SearchHelper,
        *,
        force_init: bool = False,
        force_correct: bool | Literal["add"] = False,
        force_search: bool = False,
        force_retrieve: bool = False,
        force_query: bool = False,
        skip_query: bool = False,
        retrieval_filters: Sequence[CandidateFilter] | None = None,
        final_filters: Sequence[CandidateFilter] | None = None,
    ) -> None:
        logger.info("Start processing tables.")

        # Step 0: 预预处理
        for t in self.db.auto_flush(tqdm(self.tasks, colour="#B0E0E6")):
            if t.init_process(force=force_init):
                self.save_task(t)
        logger.info("Preliminary process completed.")

        # Step 1: 纠错
        # warning: 如果表比较多，这个列表会很大！后面也是一样
        entities_to_search = (cell.value for task in self.tasks for col in task.table if col.searchable for cell in col)
        c = SpellChecker.spell_check_process(entities_to_search, make_list(spell_checkers), force=force_correct)
        c |= set(entities_to_search)

        # Step 2: 找实体
        searchers_ = make_list(searchers)
        for searcher in searchers_:
            searcher.search_store(c, force=force_search)
        # 有个问题  IC 169305167，整体搜是搜不到的，但搜后面那个数可以搜到

        # TODO 现在还没有考虑不同KG的数据问题

        # Step 3: 回填
        logger.info("Start back-patching.")
        if retrieval_filters is None:
            retrieval_filters = [F.order_by_rank, F.limiter(50)]
        if searchers:
            logger.info("Start forward querying.")
            qids = flatten(task.retrieve_data_fwd(searchers_[0]) for task in self.tasks)
            EntityManager.store_wikidata_entities(qids, force=force_query, with_properties=False)
            for t in self.tasks:
                t.retrieve_data(searchers_[0], force=force_retrieve, filters=retrieval_filters)
                self.save_task(t)
            self.db.flush()

        # Step 4: 找详细数据
        if not skip_query:
            qids = (cand.qid for task in self.tasks for col in task.table for cell in col for cand in cell)
            ancestor_level = config.get_("process.entity.ancestor_level", 3)
            EntityManager.store_wikidata_entities(qids, force=force_query, level=ancestor_level)

        logger.info("Start finalizing.")
        if final_filters is None:
            final_filters = [
                F.score_by_ratio(fuzz.ratio),
                F.order_by(key=lambda c: -c.score / (1 + c.rank) ** 0.25),
            ]
        for t in tqdm(self.tasks, colour="#ADD8E6"):
            t.finalize(force=True, filters=final_filters)
            self.save_task(t)
        self.db.flush()
        logger.info("Process completed.")
