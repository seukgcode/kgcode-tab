from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.jsonio import read_json
from src.process import dbhelper

# corr: dict[str, set[str]] = {}

qids = []

for p in Path(r"E:\Project\kg-tab\data\HT-Valid-mixed\spell_check_json").iterdir():
    data = read_json(p)
    for c in data["data"]:
        if not c["canSearch"]:
            continue
        for d in c["column"]:
            if not d["QIDs"] or not d["value"]:
                continue
            qids.extend(d["QIDs"])
            assert dbhelper.db["wikidata_search"].count_where("key = ?", [d["value"].lower()]) > 0
            assert dbhelper.db["correction"].count_where("key = ?", [d["value"]]) > 0
            assert dbhelper.db["correction"].count_where("value = ?", [d["value"].lower()]) > 0
            # dbhelper.add_wd_search([d["value"].lower()], [[{"id": x, "match": "other"} for x in d["QIDs"]]])

print("Total:", len(qids))
cnt = 0
failures = []
for q in qids:
    if dbhelper.db["wikidata_entity"].count_where("qid = ?", [q]) == 0:
        if dbhelper.db["wikidata_search"].count_where("id = ?", [q]) == 0:
            # if dbhelper.db["wikidata_search"].count_where("id = ?", [q]) != 0:
            #     print("????")
            # yyy = list(dbhelper.db["wikidata_search"].rows_where("id = ?", [q], select="key"))
            # if len(yyy) > 0:
            #     # print(yyy, q)
            #     print(list(dbhelper.db["correction"].rows_where("value = ?", [yyy[0]["key"]])))
            cnt += 1
            failures.append(q)
print(cnt)
print(failures)

# 问题在于，search里面有，但是没存下去，或者根本就没有搜它