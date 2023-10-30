import sys
from pathlib import Path
from pprint import pprint
from time import time

from sqlite_utils import Database

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.jsonio import read_json

db = Database(".cache/tab.db", recreate=False)


def create_entity():

    data = read_json(".cache/wikidata-entities.json")
    # entity = db.create_table("entity", {"key": str, "value": str, "prop": bool}, pk="key")
    # entity.insert_all([{"key": x["qid"], "value": x, "prop": "properties" in x} for x in data.values()])
    # print(entity.count_where("prop"))
    # print(list(db.query("select * from entity limit 1")))
    table = db.create_table(
        "wikidata_entity",
        {"qid": str, "label": str, "description": str, "aliases": str, "properties": str},
        pk="qid",
    )
    table.insert_all(list(data.values()))
    print(table.count_where("properties is not null"))
    print(list(db.query("select * from wikidata_entity limit 1")))


def create_search():
    data = read_json(".cache/wikidata-search.json")
    db["wikidata_search"].drop()
    table = db.create_table("wikidata_search", {"key": str, "id": str, "match": str}, pk=["key", "id"])
    table.insert_all([{"key": k, "id": x["id"], "match": x["match"]} for k, v in data.items() for x in v])


def create_search_index():
    data = read_json(".cache/wikidata-search.json")
    db["wikidata_search_idx"].drop()
    table = db.create_table("wikidata_search_idx", {"key": str}, pk="key")
    table.insert_all([{"key": k} for k in data.keys()])


def create_correction():
    data = read_json(".cache/corrections.json")
    table = db.create_table("correction", {"key": str, "value": str}, pk=["key", "value"])
    table.insert_all([{"key": k, "value": x} for k, v in data.items() for x in v])


# create_entity()
# create_search()
# create_search_index()
# create_correction()
# print(
#     db.execute("""
# select label
# from wikidata_entity
# inner join wikidata_search on qid = id
# where key = "poisson"
# """).fetchall()
# )
print(db["wikidata_entity"].count_where("properties is null"))
# print([
#     a[1] for p in db["wikidata_entity"].rows_where(
#         "qid in ('Q5') and properties is not null",
#         select="properties",
#     ) for k, v in json.loads(p["properties"]).items() for a in v if a[0] == "wikibase-entityid"
# ])
# print(db["wikidata_entity"].get("Q76498724"))
# print(list(db["correction"].rows_where("key = 'Hua Hin'")))
print(len({x["value"] for x in db.query("select value from correction")}))
print(len({x["key"] for x in db.query("select key from wikidata_search_idx")}))
print(
    len({
        x["value"]
        for x in db.query("select value from correction where value not in (select key from wikidata_search_idx)")
    })
)
print(
    len({x["id"]
         for x in db.query("select id from wikidata_search where id not in (select qid from wikidata_entity)")})
)
# print(list(db.query("select * from wikidata_search where key='leda 1245565'")))


def test():

    t1 = time()
    for i in range(10000):
        db.execute(
            """
        select E.qid, E.label, E.description
        from wikidata_entity E
        where E.qid in (
            select S.id
            from wikidata_search S
            where S.key in (
                select C.value
                from correction C
                where C.key = ?
            )
        )
    """, ["Gliese 818"]
        ).fetchall()
    t2 = time()
    for i in range(10000):
        db.execute(
            """
        select E.qid, E.label, E.description
        from wikidata_entity E
        where E.qid in (
            select S.id
            from correction C
            inner join wikidata_search S on C.value = S.key
            where C.key=?
        )
    """, ["Gliese 818"]
        ).fetchall()
    t3 = time()
    for i in range(10000):
        list(
            db.query(
                """
        select E.qid, E.label, E.description
        from correction C
        inner join wikidata_search S on C.value = S.key
        inner join wikidata_entity E on S.id = E.qid
        where C.key=?
    """, ["Gliese 818"]
            )
        )
    t4 = time()
    print(t2 - t1, t3 - t2, t4 - t3)
    # join 好像比 in 效率高
    pprint(
        list(
            db.query(
                """
        select E.qid, E.label, E.description
        from correction C
        inner join wikidata_search S on C.value = S.key
        inner join wikidata_entity E on S.id = E.qid
        where C.key=?
    """, ["Gliese 818"]
            )
        ),
        width=120
    )
    pprint(type(db["wikidata_entity"].get("Q5")["properties"]))