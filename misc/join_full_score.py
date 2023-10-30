import pandas as pd

# df1 = pd.read_csv(r"D:\Tencent Files\2380433991\FileRecv\wrong_cases_primary.csv")
df1 = pd.read_csv(r"D:\Tencent Files\2380433991\FileRecv\wrong_cases.csv")
# df2 = pd.read_csv(r"E:\Project\tmp\full_score.csv")
df2 = pd.read_csv(r"E:\Project\seuKG\.result\history\20230810 235629\full_score.csv")

ddd = set[str]()
for row in df1.itertuples(index=False):
    ddd.add('|'.join(map(str, row[: 3])))

# df = df1.join(df2, on=["table", "row", "col"], how="left")
# df.to_csv(r"E:\Project\tmp\full_score_primary_err.csv")

# df1["id"] = df1["table"].str + df1["row"].str

res = []
for row in df2.itertuples(index=False):
    if '|'.join(map(str, row[: 3])) in ddd:
        res.append(row)

pd.DataFrame(res).to_csv(r"E:\Project\tmp\full_score_limaye_primary_err.csv")