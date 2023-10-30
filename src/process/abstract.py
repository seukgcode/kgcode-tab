from __future__ import annotations

from typing import Any, cast

from ..table.entity import Entity
from ..table.table_data import Table
from ..utils.jsonio import read_json
from ..utils.lists import is_list_of
from .wikibase import EntityManager


def make_row_abstract(table: Table, j: int):
    return "; ".join(f"{col.name} is {col[j].text}" if col.name else col[j].text for col in table)


def make_abstract_from_label_description(entity: Entity):
    return f"{entity.label}: {entity.description}" if entity.description else entity.label


def from_description_instanceof(entity: Entity):
    assert entity.properties is not None
    s = ", ".join((EntityManager.get(p[1]).label for p in entity.properties.get("P31", [])))
    return f"{entity.description or entity.label}. Instance of {s}."


def _desc_prop(plist: list[Any]):
    if plist[0] == "wikibase-entityid":
        return EntityManager.get(plist[1]).label or ""
    return ", ".join(map(str, plist[1:]))


def _desc_props(plists: list[list[Any]]):
    return "; ".join(map(_desc_prop, plists[:3]))


def make_abstract_from_given_properties(
    qids: list[str] | list[Entity], scope: set[str], props_dict: dict[str, str] | None = None
):
    """根据属性生成描述列表"""
    _props_dict = props_dict or read_json("data/prop_dict.json")
    res: list[str] = []
    if not qids:
        return res
    if is_list_of(qids, str):
        entities = EntityManager.gets(cast(list[str], qids), force=True)
    else:
        entities = cast(list[Entity], qids)

    for e in entities:
        assert e.properties is not None
        if e.properties is not None:
            pp = "[SEP]".join(
                f" {_props_dict[k]} [PROP] {_desc_props(v)} "
                for k, v in e.properties.items()
                if k in scope and k in _props_dict
            )
            res.append(f"[CLS] {make_abstract_from_label_description(e)} [SEP] {pp}")
    return res


def make_abstract():
    ...
