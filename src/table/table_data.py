from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, TypeAlias, overload

from .entity import Entity

SpacyType: TypeAlias = Literal[
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]

excl = set(Path("data/exclude.txt").read_text().splitlines())


class WDValueType:
    WikibaseEntityId: TypeAlias = tuple[Literal["wikibase-entityid"], str, str]
    GlobeCoordinate: TypeAlias = tuple[Literal["wikibase-globecoordinate"], float, float]
    Time: TypeAlias = tuple[Literal["time"], str]
    String: TypeAlias = tuple[Literal["string"], str]
    MonolingualText: TypeAlias = tuple[Literal["monolingualtext"], str, str]
    Quantity: TypeAlias = tuple[Literal["quantity"], str]


PropertyList: TypeAlias = (
    list[WDValueType.WikibaseEntityId]
    | list[WDValueType.GlobeCoordinate]
    | list[WDValueType.Time]
    | list[WDValueType.String]
    | list[WDValueType.MonolingualText]
    | list[WDValueType.Quantity]
)


@dataclass
class Candidate:
    qid: str
    match: str
    rank: int
    score: float
    st_score: float = 1
    cell: Cell | None = None
    metadata: Any = None
    _entity: Entity | None = None
    _type_ancestors: list[tuple[int, str]] | None = None
    _type_ancestor_hint: int = -1

    @property
    def entity(self) -> Entity:
        from ..process.wikibase import EntityManager

        if not self._entity:
            self._entity = EntityManager.get(self.qid)
        return self._entity

    @property
    def entity_p(self) -> Entity:
        from ..process.wikibase import EntityManager

        if not self._entity or self._entity.properties is None:
            self._entity = EntityManager.get(self.qid, force=True)
        return self._entity

    def iter_typeid(self) -> Iterable[str]:
        # TODO: for DBPedia
        assert self.entity_p.properties is not None
        return (p[1] for p in self.entity_p.properties.get("P31", []))

    def iter_type_ancestors(self, depth: int = 3) -> Iterable[tuple[int, str]]:
        if self._type_ancestors is None or depth != self._type_ancestor_hint:
            self._type_ancestor_hint = depth
            self._type_ancestors = list(self.__iter_type_ancestors(depth))
        yield from self._type_ancestors

    def __iter_type_ancestors(self, depth: int) -> Iterable[tuple[int, str]]:
        from ..process.wikibase import EntityManager

        assert self.entity_p.properties is not None
        classes: list[str] = [p[1] for p in self.entity_p.properties.get("P31", [])]
        yield from ((0, x) for x in classes)
        for i in range(1, depth + 1):
            classes = list(chain.from_iterable(EntityManager.get_subclasses(c) for c in classes))
            yield from ((i, x) for x in classes)

    def iter_entity_props(self) -> Iterable[str]:
        props = self.entity_p.properties
        assert props is not None
        # return (k for k in props.keys() if k not in excl)
        return (f"{k}={p[1]}" for k, v in props.items() if v and v[0][0] == "wikibase-entityid" for p in v)

    def iter_entity_prop_groups(self) -> Iterable[tuple[str, list[str]]]:
        props = self.entity_p.properties
        assert props is not None
        return ((k, [p[1] for p in v]) for k, v in props.items() if v and v[0][0] == "wikibase-entityid")

    def to_dict(self):
        return {
            "qid": self.qid,
            "match": self.match,
            "rank": self.rank,
            "score": self.score,
            "st_score": self.st_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, kvs: dict):
        return cls(
            qid=kvs["qid"],
            match=kvs["match"],
            rank=kvs["rank"],
            score=kvs["score"],
            st_score=kvs.get("st_score", 1),
            metadata=kvs.get("metadata"),
        )

    def __hash__(self):
        return hash(self.qid)


@dataclass
class Cell:
    is_none: bool  # 如果是表示这个cell是否为空，
    text: str
    value: str
    corrections: list[str]
    candidates: list[Candidate] = field(default_factory=lambda: [])
    candidates_ini: list[Candidate] = field(default_factory=lambda: [])  # 初始候选项
    metadata: Any = None

    def backref(self):
        for ca in self.candidates:
            ca.cell = self

    def __getitem__(self, k: int) -> Candidate:
        # if not self.candidates:
        #     raise RuntimeError("The cell is not searchable!")
        return self.candidates[k]

    def __iter__(self):
        yield from self.candidates

    def to_dict(self):
        return {
            "is_none": self.is_none,
            "text": self.text,
            "value": self.value,
            "corrections": self.corrections,
            "candidates": list(map(Candidate.to_dict, self.candidates)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, kvs: dict):
        candidates = [Candidate.from_dict(a) for a in kvs["candidates"]]
        cell = cls(
            is_none=kvs["is_none"],
            text=kvs.get("text", kvs["value"]),
            value=kvs["value"],
            corrections=kvs["corrections"],
            candidates=candidates,
            candidates_ini=candidates,
            metadata=kvs.get("metadata"),
        )
        cell.backref()
        return cell


@dataclass
class Column:
    empty: bool  # 是否为空
    searchable: bool  # 等价于为非数值类型
    numerical: bool
    keyable: bool = False
    type: SpacyType | None = None
    name: str | None = None
    cells: list[Cell] = field(default_factory=lambda: [])
    metadata: Any = None

    def __getitem__(self, row: int) -> Cell:
        return self.cells[row]

    def __iter__(self):
        yield from self.cells

    @property
    def cell_texts(self) -> list[str]:
        return [c.value for c in self.cells]

    def iter_candidates(self) -> Iterable[Candidate]:
        return (cand for cell in self for cand in cell)

    def to_dict(self):
        return {
            "empty": self.empty,
            "searchable": self.searchable,
            "numerical": self.numerical,
            "keyable": self.keyable,
            "type": self.type,
            "name": self.name,
            "cells": list(map(Cell.to_dict, self.cells)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, kvs: dict):
        return cls(
            empty=kvs.get("empty", False),
            searchable=kvs["searchable"],
            numerical=kvs.get("numerical", False),
            keyable=kvs.get("keyable", False),
            type=kvs["type"],
            name=kvs["name"],
            cells=[Cell.from_dict(a) for a in kvs["cells"]],
            metadata=kvs.get("metadata"),
        )


@dataclass
class Table:
    path: Optional[str]
    name: str
    """Name of the table"""
    rows: int
    """Number of rows"""
    cols: int
    """Number of columns"""
    columns: list[Column]
    """Column data"""
    key_col: int = 0
    row_texts: list[str] = field(default_factory=lambda: [])
    """Index of key/subject column"""
    searchable: bool = False
    """Any column is searchable"""
    processed: bool = False
    retrieved: bool = False
    completed: bool = False
    """Whether this table is processed"""
    metadata: Any = None

    @property
    def sub_col(self) -> Column:
        return self.columns[self.key_col]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def iter_cells(self) -> Iterable[Cell]:
        return (cell for col in self for cell in col)

    def iter_candidates(self) -> Iterable[Candidate]:
        return (cand for col in self for cell in col for cand in cell)

    @overload
    def __getitem__(self, col_row: int) -> Column:
        ...

    @overload
    def __getitem__(self, col_row: tuple[int, int]) -> Cell:
        ...

    def __getitem__(self, col_row: int | tuple[int, int]) -> Column | Cell:
        if isinstance(col_row, int):
            return self.columns[col_row]
        return self.columns[col_row[0]][col_row[1]]

    def __iter__(self):
        yield from self.columns

    def to_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "columns": list(map(Column.to_dict, self.columns)),
            "key_col": self.key_col,
            "row_texts": self.row_texts,
            "searchable": self.searchable,
            "processed": self.processed,
            "retrieved": self.retrieved,
            "completed": self.completed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, kvs: dict):
        columns = [Column.from_dict(a) for a in kvs["columns"]]
        row_texts = kvs.get("row_texts", [])
        if not row_texts:
            row_texts = ["; ".join(col[i].text for col in columns) for i in range(len(columns[0].cells))]
        return cls(
            path=kvs.get("path"),
            name=kvs["name"],
            rows=kvs["rows"],
            cols=kvs["cols"],
            columns=[Column.from_dict(a) for a in kvs["columns"]],
            key_col=kvs["key_col"],
            row_texts=row_texts,
            searchable=kvs["searchable"],
            processed=kvs["processed"],
            retrieved=kvs["retrieved"],
            completed=kvs["completed"],
            metadata=kvs.get("metadata"),
        )
