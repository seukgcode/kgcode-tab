import json
import os
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import jinja2
import pandas as pd
import plotly as py
import plotly.express as px

from ...utils import PathLike, read_json


def parse_datetime(s: str):
    dt = list(map(int, [s[: 4], s[4 : 6], s[6 : 8], s[9 : 11], s[11 : 13], s[13 : 15]]))
    return datetime(dt[0], dt[1], dt[2], dt[3], dt[4], dt[5])


def _get_reports(path: Path):
    reports = []
    for p in path.iterdir():
        if p.is_dir() and (pp := p / "report.json").is_file():
            try:
                rep = read_json(pp)
            except Exception:
                continue
            if "datetime" not in rep:
                rep["datetime"] = datetime.strftime(parse_datetime(p.stem), "%Y-%m-%d %H:%M:%S.%f")
            if "dataset" not in rep:
                rep["dataset"] = "Unknown"
            reports.append(rep)
    reports.sort(key=itemgetter("datetime"))
    return reports


def _generate_html(reports: list, by_time: bool):
    report_data = []
    cnt = defaultdict[str, int](int)
    for rep in reports:
        ds_name = rep["dataset"].removesuffix("Dataset")
        dat = {
            "index": cnt[ds_name],
            "datetime": datetime.strptime(rep["datetime"], "%Y-%m-%d %H:%M:%S.%f"),
            "dataset": ds_name,
            "params": "".join(f"<br>  {k}: {v}" for k, v in rep["parameters"].items()) if "parameters" in rep else "",
            "metadata": json.dumps(rep.get("metadata")),
        }
        cnt[ds_name] += 1
        report_data.extend(
            dat | {
                "task": t, "metric": m, "score": rep[t][m], "values":
                f'{rep[t]["correct"]}/{rep[t]["annotated"]}/{rep[t]["total"]}'
            } for t in ("CTA", "CEA", "CPA") if t in rep for m in ("F1", "P", "R")
        )
    report_df = pd.DataFrame(report_data)
    charts = []
    for i, (ds, df) in enumerate(report_df.groupby("dataset"), 1):
        fig = px.line(
            df,
            x="datetime" if by_time else "index",
            y="score",
            color="metric",
            facet_row="task",
            title=ds,
            markers=True,
            height=700,
            hover_data=["values", "metadata", "params"] + ([] if by_time else ["datetime"]),
        )
        # fig.xa.update_xaxes(rangeslider_visible=True)
        chart = py.offline.plot(fig, include_plotlyjs=False, output_type="div")
        charts.append(chart)
    return charts


def generate_report(result_path: PathLike, by_time: bool = False, auto_open: bool = True):
    result_path = Path(result_path)
    reports = _get_reports(result_path / "history")
    charts = _generate_html(reports, by_time)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=Path(__file__).parent))
    template = env.get_template("template.jinja")
    html = template.render(charts=charts)
    output_path = result_path / "report.html"
    output_path.write_text(html)
    if auto_open:
        os.startfile(output_path)