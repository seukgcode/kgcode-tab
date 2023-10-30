from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.jsonio import read_json
from src.process import dbhelper

# corr: dict[str, set[str]] = {}

for p in Path(r"E:\Project\kg-tab\data\HT-Valid-mixed\spell_check_json").iterdir():
    data = read_json(p)
    for c in data["data"]:
        if not c["canSearch"]:
            continue
        for d in c["column"]:
            if not d["correction"] or not d["value"]:
                continue
            cccc = {s.lower() for s in d["correction"]}
            # corr[d["value"]] = cccc
            # assert dbhelper.db.query("")
            dbhelper.add_correction(d["value"], list(cccc))
    # break
# print(corr)
