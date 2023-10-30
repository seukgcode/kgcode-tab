from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.jsonio import read_json
from src.process import dbhelper
from src.process.preprocess import TableProcessor

df = pd.read_csv(r"E:\Project\kg-tab\data\HT-Valid-mixed\DataSets\Valid\gt\cea_gt.csv", header=None)
tp = TableProcessor(".cache/sample-tables.json")
tp.add_tables(r"E:\Project\kg-tab\data\HT-Valid-mixed\DataSets\Valid\tables")
tables = {t.table.name: t.table for t in tp.tasks}
cnt = 0
for idx, row in df.iterrows():
    table = tables[row[0]]
    q = row[3].split("/")[-1]
    r = row[1] - 1
    c = row[2]
    if len(list(filter(lambda x: x.qid == q, table[c, r].candidates))) == 0:
        cnt += 1
        print(row.tolist(), dbhelper.db["wikidata_entity"].count_where("qid = ?", [q]))
print(cnt)