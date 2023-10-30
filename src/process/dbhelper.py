from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable

import orjson as json
from sqlite_utils import Database
from sqlite_utils.db import Table

from ..utils import PathLike, config, logger

logger.info("Load db: %s", config["process"]["db_path"])

Path(config["process"]["db_path"]).parent.mkdir(exist_ok=True)

db = Database(config["process"]["db_path"], recreate=False)


def _get_table(name: str, columns: dict, pk: list[str] | str) -> Table:
    table = db[name]
    assert type(table) == Table
    if not table.exists():
        table = db.create_table(name, columns, pk=pk)
        logger.info(f"Create table {name}")
    return table


def _ensure_column(tab: Table, col_name: str, col_type: Any | None = None, not_null_default: Any | None = None):
    if col_name not in tab.columns_dict:
        tab.add_column(col_name, col_type, not_null_default=not_null_default)
        return True
    return False


_db_correction = _get_table("correction", {"key": str, "value": str, "similarity": float}, pk=["key", "value"])
_db_correction_failure = _get_table("correction_failure", {"key": str}, pk="key")
_db_wikidata_search = _get_table(
    "wikidata_search", {"key": str, "id": str, "match": str, "rank": int}, pk=["key", "id"]
)
_db_wikidata_search_idx = _get_table("wikidata_search_idx", {"key": str}, pk="key")
_db_wikidata_entity = _get_table(
    "wikidata_entity", {"qid": str, "label": str, "description": str, "aliases": str, "properties": str}, pk="qid"
)


def _update():
    if _ensure_column(_db_correction, "similarity", float, 1.0):
        from rapidfuzz import fuzz

        items = ({**t, "similarity": fuzz.partial_ratio(t["key"], t["value"])} for t in _db_correction.rows)
        _db_correction.upsert_all(items, pk=["key", "value"])  # type: ignore

    _ensure_column(_db_wikidata_search, "rank", int, -1)
    db.execute(
        """
    update wikidata_search
    set rank = -1
    where key in (
        select key
        from wikidata_search
        group by key, match
        having count(distinct id) > 1 and max(rank) == 0
    )
    """
    )
    db.conn.commit()
    _db_wikidata_search_idx.delete_where("key in (select key from wikidata_search where rank == -1)")
    _db_wikidata_search.delete_where("rank == -1")


# _update()


def _L(given: Iterable[str]) -> str:
    return "(" + ",".join(f""""{x.replace('"', '""')}\"""" for x in set(given)) + ")"


def _S(given: list[str]) -> str:
    """生成占位符"""
    return f"({','.join(['?'] * len(given))})"


def get_keys_in_table(table: str, key: str, given: Iterable[str], additional: str = "") -> set[str]:
    # 找given中不在db中的
    where_condition = f"{key} in {_L(given)}"  # 用占位符会有变量过多的问题
    if additional:
        where_condition = f"({additional}) and ({where_condition})"
    sql = f"""select {key}
              from {table}
              where {where_condition}"""
    return set(map(itemgetter(0), db.execute(sql)))


def remove_keys_in_table(table: str, key: str, given: Iterable[str], additional: str = ""):
    where_condition = f"{key} in {_L(given)}"  # 用占位符会有变量过多的问题
    if additional:
        where_condition = f"({additional}) and ({where_condition})"
    db[table].delete_where(where_condition)  # type: ignore


def remove_correction_keys(given: Iterable[str]):
    remove_keys_in_table("correction", "key", given)


def get_correction_keys(given: Iterable[str]) -> set[str]:
    return (
        set(given)
        - get_keys_in_table("correction", "key", given)
        - get_keys_in_table("correction_failure", "key", given)
    )


def add_correction(key: str, values: Iterable[str]) -> None:
    from rapidfuzz import fuzz

    _db_correction.upsert_all(
        ({"key": key, "value": v, "similarity": fuzz.partial_ratio(key, v)} for v in values), pk=["key", "value"]
    )  # type: ignore


# def add_identity_corrections(values: Iterable[str]) -> None:
#     _db_correction.upsert_all(({"key": v, "value": v} for v in values), pk=["key", "value"])  # type: ignore


def add_correction_failures(values: Iterable[str]):
    _db_correction_failure.upsert_all(({"key": v} for v in values), pk="key")  # type: ignore


def get_correction(key: str) -> Iterable[str]:
    sql = """select value
             from correction
             where key = ?"""
    return map(itemgetter(0), db.execute(sql, (key,)))


def get_corrections(keys: Iterable[str]) -> Iterable[str]:
    # 获取一组文本的所有纠错结果
    sql = f"""select distinct value
              from correction
              where key in {_L(keys)}"""
    return map(itemgetter(0), db.execute(sql))


def get_wd_search_keys(given: Iterable[str], check: bool = False) -> set[str]:
    # check: 包含没有搜到的
    if check:
        keys_in_db = get_keys_in_table("wikidata_search", "key", given, "rank != -1")
    else:
        keys_in_db = get_keys_in_table("wikidata_search_idx", "key", given)
    return set(given) - keys_in_db


def remove_wd_search_keys(given: Iterable[str]) -> None:
    remove_keys_in_table("wikidata_search", "key", given)


def add_wd_search(search: list[str], res: list[list[dict]]) -> None:
    _db_wikidata_search_idx.upsert_all([{"key": k} for k in search], pk="key")  # type: ignore
    _db_wikidata_search.upsert_all(
        (
            {"key": k, "id": x["id"], "match": x["match"], "rank": i}
            for k, v in zip(search, res)
            for i, x in enumerate(v)
        ),
        pk=["key", "id"],
    )  # type: ignore


def get_wd_search(key: str | Iterable[str]) -> Iterable[tuple[str, str, int]]:
    if isinstance(key, str):
        sql = """select id, match, rank
                 from wikidata_search
                 where key = ?"""
        return db.execute(sql, (key,))
    else:
        sql = f"""select id, match, min(rank)
                  from wikidata_search
                  where key in {_L(key)}
                  group by id, match"""
        return db.execute(sql)


def get_wd_entity_keys(given: Iterable[str], with_properties: bool) -> set[str]:
    if with_properties:
        keys_in_db = get_keys_in_table("wikidata_entity", "qid", given, "label is not null and properties is not null")
    else:
        keys_in_db = get_keys_in_table("wikidata_entity", "qid", given, "label is not null")
    return set(given) - keys_in_db


def add_wd_entities(ids: list[str], res: dict[str, list]) -> None:
    _db_wikidata_entity.upsert_all(
        (
            {  # type: ignore
                "qid": ids[i],
                "label": res["labels/en"][i] or "",
                "description": res["descriptions/en"][i] or "",
                "aliases": res["aliases/en"][i] or [],
                "properties": res["properties"][i] or {} if "properties" in res else None,
            }
            for i in range(len(ids))
        ),
        pk="qid",
    )  # type: ignore


def get_wd_entity(qid: str) -> tuple[str, str, str, str, str]:
    return db.execute("select * from wikidata_entity where qid = ?", (qid,)).fetchone()


def get_wd_ids_in_properties(ids: Iterable[str]) -> set[str]:
    sql = f"""select properties
              from wikidata_entity
              where properties is not null and qid in {_L(ids)}"""
    return {
        a[1] for p in db.execute(sql) for k, v in json.loads(p[0]).items() for a in v if a[0] == "wikibase-entityid"
    }


def get_wd_ids_of_P31(ids: Iterable[str]) -> set[str]:
    sql = f"""select properties
              from wikidata_entity
              where properties is not null and qid in {_L(ids)}"""
    return {
        a[1]
        for p in db.execute(sql)
        for k, v in json.loads(p[0]).items()
        if k in ("P31", "P279")
        for a in v
        if a[0] == "wikibase-entityid"
    }


def merge_from(path: PathLike) -> None:
    db.attach("other", str(path))
    for row in db.execute("select * from other.sqlite_master where type = 'table'"):
        print(row)
        db.execute(f"insert or ignore into {row[1]} select * from other.{row[1]}")
    db.conn.commit()
    db.execute("detach database other")
    logger.info(f"Successfully merge from {path}")
