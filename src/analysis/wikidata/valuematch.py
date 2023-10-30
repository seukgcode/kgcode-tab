import re
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterable, TypeAlias

from dataclasses_json import DataClassJsonMixin
from more_itertools import flatten
from rapidfuzz import fuzz

from ...process import EntityManager
from ...table import Cell, Table, WDValueType
from ..utils import make_list2, make_list3

opt_str: TypeAlias = str | None

months = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
months_abbr = [s[:3] for s in months]

float_pat = re.compile(r"[^0-9.%-]")

lat_pat = re.compile(r"(\d+\.?\d*)°((\d+)['′]((\d+)″)?)?(N|S)")
lon_pat = re.compile(r"(\d+\.?\d*)°((\d+)['′]((\d+)″)?)?(E|W)")

time_pat_num0 = re.compile(r"\d+")
time_pat_num = re.compile(r"[1-9]\d*")  # number
time_pat_dec = re.compile(r"\d+\.(0\d+|[1-9]\d*)")  # decimal
time_pat_dsd = re.compile(r"\d+\.\d+\.\d+")  # dot separated date
time_pat_word = re.compile(r"[a-z]+")  # word
time_pat_csn = re.compile(r"\d+,\d\d\d")  # comma separated number


def try_parse_float(x: str) -> float | None:
    x = x.replace("$", "").replace("£", "").replace(",", "")
    try:
        x = float_pat.sub(" ", x).strip()
        return float(x[:-1]) / 100 if x.endswith("%") else float(x)
    except ValueError:
        return None


@dataclass
class SimilarityMeta(DataClassJsonMixin):
    avg: float
    bests: list[float]
    pids: list[str | None]
    qids: list[str | None]
    labels: list[str | None]
    matches: list[list[tuple[str, str, str | None, float]]]  # (p, l, e, s)

    @classmethod
    def new(cls, cols: int):
        return cls(
            avg=0,
            bests=[0] * cols,
            pids=[None] * cols,
            qids=[None] * cols,
            labels=[None] * cols,
            matches=[[] for _ in range(cols)],
        )


def quick_similarity(table: Table):
    """Get a readable format of similarity meta

    Args:
        table (Table): A table

    Returns:
        _type_: _description_
    """
    result = []
    for cell in table.sub_col:
        metas: list[SimilarityMeta] = [c.metadata for c in cell]  # type: ignore
        result.append(
            {
                "avg": [m.avg for m in metas],
                "bests": [m.bests for m in metas],
                "pids": [m.pids for m in metas],
                "qids": [m.qids for m in metas],
                "labels": [m.labels for m in metas],
            }
        )
    return result


def cutoff(x: float, a: float):
    # return x if x >= a else 0
    return max((x - a + 1e-6) / (1 - a + 1e-6), 0)


def is_all_num(s: str) -> bool:
    return all("0" <= c <= "9" or c == "." or c == " " for c in s)


class Matcher:
    alpha: float = 0.9
    beta: float = 0.98
    beta2: float = 0.5

    @classmethod
    def num_similarity(cls, a: float, b: float) -> float:
        d = abs(a - b) / max(abs(a) + 1e-9, abs(b) + 1e-9)
        return cutoff(1 - d, Matcher.beta)

    @classmethod
    def num_similarity_abs(cls, a: float, b: float) -> float:
        return cls.num_similarity(a, b)
        d = abs(a - b)
        return cutoff(1 - d, Matcher.beta2)

    @classmethod
    def match_some(cls, src: Iterable[tuple[str, float]], cutoff: float = 0) -> tuple[float, opt_str]:
        max_sim = 0.0
        max_label = None
        for l, s in src:
            if s >= cutoff and s > max_sim:
                max_sim = s
                max_label = l
                if max_sim == 1.0:
                    break
        return max_sim, max_label

    @classmethod
    def match_string(cls, tab: str, src: Iterable[str]) -> tuple[float, opt_str]:
        return cls.match_some(((s, fuzz.ratio(tab, s.lower()) / 100) for s in src), cls.alpha)

    @classmethod
    def wikibase_entity_id(
        cls, tab: str, properties: list[WDValueType.WikibaseEntityId]
    ) -> tuple[float, opt_str, opt_str]:
        max_sim = 0.0
        max_qid = None
        max_label = None
        # is_num = is_all_num(tab)
        for p in properties:
            if e := EntityManager.get(p[1]):
                for a in e.iter_aliases():
                    # if is_num != is_all_num(a):
                    #     continue
                    current_sim = fuzz.ratio(tab, a.lower()) / 100
                    if current_sim >= cls.alpha and current_sim > max_sim:
                        max_sim = current_sim
                        max_qid = str(p[1])
                        max_label = a
                        if max_sim == 1.0:
                            break
                if max_sim == 1.0:
                    break
        return max_sim, max_qid, max_label

    @classmethod
    def globe_coordinate(cls, tab_: str, properties: list[WDValueType.GlobeCoordinate]) -> tuple[float, opt_str]:
        max_sim = 0.0
        max_label = None
        if "°" in tab_:
            if m := lat_pat.search(tab_):
                la = (float(m[1]) + float(m[3] or 0) / 60 + float(m[5] or 0) / 3600) * (1 if m[6] == "N" else -1)
                return cls.match_some((str(p[1]), cls.num_similarity_abs(la, p[1])) for p in properties)
            elif m := lon_pat.search(tab_):
                lo = (float(m[1]) + float(m[3] or 0) / 60 + float(m[5] or 0) / 3600) * (1 if m[6] == "E" else -1)
                return cls.match_some((str(p[2]), cls.num_similarity_abs(lo, p[2])) for p in properties)
        elif (tab := try_parse_float(tab_)) is not None:
            for p in properties:
                if p[1]:  # latitude
                    current_sim = cls.num_similarity_abs(tab, float(p[1]))
                    if current_sim > max_sim:
                        max_sim = current_sim
                        max_label = str(p[1])
                if p[2]:  # longitude
                    current_sim = cls.num_similarity_abs(tab, float(p[2]))
                    if current_sim > max_sim:
                        max_sim = current_sim
                        max_label = str(p[2])
                if max_sim == 1.0:
                    break
        return max_sim, max_label

    @classmethod
    def time_sim(cls, pt: str, words: list[str], nums: list[int]) -> float:
        l3 = nums[:]
        _y, _m, _d = list(map(int, time_pat_num0.findall(pt)[:3]))  # 年月日
        _mi = int(_m) - 1
        sim = 0
        if _mi >= 0 and (months[_mi] in words or months_abbr[_mi] in words):  # 有月份单词
            if len(nums) == 0:  # 只有月
                sim = 1
            elif len(nums) == 1:  # 有年或日
                sim = 1 if _y in nums or _d in nums else 0
            elif len(nums) == 2:  # 年月日都应该有
                if _y in nums:
                    sim = 1 if _d in nums else 0.8
        else:
            if len(nums) == 1:  # 只有一个，优先匹配年
                if _y in nums:
                    sim = 1
            elif len(nums) == 2:  # 有两个，匹配月加年或日
                if _m in nums:
                    l3.remove(_m)
                    sim = 0.4 + int(_y in l3 or _d in l3) * 0.6
            else:  # 年要匹配，月日加分
                if _y in nums:
                    sim = 0.8
                    l3.remove(_y)
                    if _m in l3:
                        sim += 0.1
                        l3.remove(_m)
                    if _d in l3:
                        sim += 0.1
                    if sim == 0.8:
                        sim = 0
        return sim

    @classmethod
    def time(cls, tab: str, properties: list[WDValueType.Time]) -> tuple[float, opt_str]:
        nums = list(map(int, time_pat_num.findall(tab)))
        if "%" in tab or time_pat_dec.search(tab) and not time_pat_dsd.search(tab):  # 筛掉小数
            return 0.0, None
        words = time_pat_word.findall(tab)
        if len(words) > 1 or time_pat_csn.search(tab):  # 筛掉有过多单词的
            return 0.0, None
        # if not l2:
        #     return max_sim, max_label
        return cls.match_some((p[1], cls.time_sim(p[1], words, nums)) for p in properties)

    @classmethod
    def string(cls, tab: str, properties: list[WDValueType.String]) -> tuple[float, opt_str]:
        return cls.match_string(tab, (p[1] for p in properties))

    @classmethod
    def monolingual_text(cls, tab: str, properties: list[WDValueType.MonolingualText]) -> tuple[float, opt_str]:
        return cls.match_string(tab, (p[1] for p in properties))

    @classmethod
    def quantity(cls, tab_: str, properties: list[WDValueType.Quantity]) -> tuple[float, opt_str]:
        if len(tab_) <= 32 and (tab := try_parse_float(tab_)) is not None:
            return cls.match_some((p[1], cls.num_similarity(tab, float(p[1]))) for p in properties)
        return 0.0, None

    @classmethod
    @property
    def params(cls):
        return {"alpha": cls.alpha, "beta": cls.beta}


class ValueMatchWikidata:
    match_cutoff: float = 0.5
    key_threshold: float = 0.8
    property_select_threshold: float = 0.15

    def __init__(self) -> None:
        ...

    def get_all_matches(self, subject: Cell, cell: Cell, cutoff: float = 0.5) -> list[tuple[str, str, str, float]]:
        result: list[tuple[str, str, str, float]] = []
        cell_label = cell.value.lower()
        if not cell_label:
            return result
        for cand in subject:
            entity = cand.entity
            assert entity.properties is not None
            for key, value in entity.properties.items():
                if not value:  # Possibly equals []
                    continue
                value_type = value[0][0]
                if value_type == "wikibase-entityid":
                    sim, qid, label = Matcher.wikibase_entity_id(cell_label, value)
                elif value_type == "globecoordinate":
                    sim, label = Matcher.globe_coordinate(cell_label, value)
                elif value_type == "time":
                    sim, label = Matcher.time(cell_label, value)
                elif value_type == "string":
                    sim, label = Matcher.string(cell_label, value)
                elif value_type == "monolingualtext":
                    sim, label = Matcher.monolingual_text(cell_label, value)
                elif value_type == "quantity":
                    sim, label = Matcher.quantity(cell_label, value)
                else:
                    continue
                if sim >= cutoff and label:
                    result.append((entity.qid, key, label, sim))
        return result

    def get_all_matches_with_entity(
        self, subject: Cell, cell: Cell, cutoff: float = 0.5
    ) -> list[tuple[str, str, str, str | None, float]]:
        from ...table.table_data import excl

        result: list[tuple[str, str, str, str | None, float]] = []
        cell_label = cell.value.lower()
        if not cell_label:
            return result
        for cand in subject:
            entity = cand.entity_p
            assert entity.properties is not None
            for key, value in entity.properties.items():
                if not value or key in excl:  # Possibly equals []
                    continue
                value_type = value[0][0]
                qid = None
                if value_type == "wikibase-entityid":
                    sim, qid, label = Matcher.wikibase_entity_id(cell_label, value)
                elif value_type == "globecoordinate":
                    sim, label = Matcher.globe_coordinate(cell_label, value)
                elif value_type == "time":
                    sim, label = Matcher.time(cell_label, value)
                elif value_type == "string":
                    sim, label = Matcher.string(cell_label, value)
                elif value_type == "monolingualtext":
                    sim, label = Matcher.monolingual_text(cell_label, value)
                elif value_type == "quantity":
                    sim, label = Matcher.quantity(cell_label, value)
                else:
                    continue
                if sim >= cutoff and label:
                    result.append((entity.qid, key, label, qid, sim))
        return result

    def get_all_matches_no_entity(
        self, subject: Cell, cell: Cell, cutoff: float = 0.5
    ) -> list[tuple[str, str, str, float]]:
        result: list[tuple[str, str, str, float]] = []
        cell_label = cell.value.lower()
        if not cell_label:
            return result
        for cand in subject:
            entity = cand.entity
            assert entity.properties is not None
            for key, value in entity.properties.items():
                if not value:  # Possibly equals []
                    continue
                value_type = value[0][0]
                if value_type == "globecoordinate":
                    sim, label = Matcher.globe_coordinate(cell_label, value)
                elif value_type == "time":
                    sim, label = Matcher.time(cell_label, value)
                elif value_type == "string":
                    sim, label = Matcher.string(cell_label, value)
                elif value_type == "monolingualtext":
                    sim, label = Matcher.monolingual_text(cell_label, value)
                elif value_type == "quantity":
                    sim, label = Matcher.quantity(cell_label, value)
                else:
                    continue
                if sim >= cutoff and label:
                    result.append((entity.qid, key, label, sim))
        return result

    def get_all_matches_entity(
        self, subject: Cell, cell: Cell, cutoff: float = 0.5
    ) -> list[tuple[str, str, str, str, float]]:
        result: list[tuple[str, str, str, str, float]] = []
        cell_label = cell.value.lower()
        for cand in subject:
            entity = cand.entity
            assert entity.properties is not None
            for key, value in entity.properties.items():
                if not value:  # Possibly equals []
                    continue
                value_type = value[0][0]
                if value_type == "wikibase-entityid":
                    sim, qid, label = Matcher.wikibase_entity_id(cell_label, value)
                    if sim >= cutoff and qid and label:
                        result.append((entity.qid, key, qid, label, sim))
        return result

    def get_match_qid(self, subject: Cell, cell: Cell, prop: str) -> str | None:
        sim_max = 0
        result = None
        cell_label = cell.value.lower()
        for cand in subject:
            entity = cand.entity
            assert entity.properties is not None
            for key, value in entity.properties.items():
                if not value or key != prop:
                    continue
                value_type = value[0][0]
                if value_type == "wikibase-entityid":
                    sim, qid, label = Matcher.wikibase_entity_id(cell_label, value)
                    if sim > sim_max:
                        sim_max = sim
                        result = qid
        return result

    def process_pro(self, table: Table, k: int, i: int, j: int, sim_metas: dict[str, SimilarityMeta]) -> None:
        """处理一个单元格

        Args:
            table (Table): _description_.
            k (int): Subject column index.
            i (int): Column index.
            j (int): Row index.
            sim_metas (dict[str, SimilarityMeta]): _description_
        """
        types = self.types
        properties = self.properties[k]
        entities = self.entities[k]
        match_dict = self.match_dict[k]

        mm = self.get_all_matches_with_entity(table[k, j], table[i, j], cutoff=self.match_cutoff)
        for q, p, l, e, s in mm:
            qp = (q, p)  # (Q, P)
            match_dict[i][j][qp] = max(match_dict[i][j].get(qp, 0), s)  # 考虑再乘上相似度
            properties[i].add(p)
            if q not in entities[j]:
                entities[j].add(q)
                types.update(e[1] for e in EntityManager.get(q).properties.get("P31", []))  # type: ignore
            sm = sim_metas[q]
            sm.matches[i].append((p, l, e, s))

    def process_rows_sub(self, table: Table, k: int):
        """在以k列为主题列的基础上处理"""
        sim_metas = [
            {ca.qid: SimilarityMeta.new(table.cols) for ca in table[k, j].candidates} for j in range(table.rows)
        ]
        for i in range(table.cols):
            if i == k:
                continue
            for j in range(table.rows):
                if "../../" in table[i, j].value:
                    continue
                self.process_pro(table, k, i, j, sim_metas[j])

        # 下面对每一列需要统计匹配p计数
        den = 0
        for i in range(table.cols):
            cnt = Counter(flatten({p for q, p in self.match_dict[k][i][j].keys()} for j in range(table.rows)))
            supported = {k for k, v in cnt.items() if v >= table.rows * ValueMatchWikidata.property_select_threshold}
            if not supported:  # 只保留匹配数量足够多的属性
                continue
            den += 1
            # max_freq = cnt.most_common(1)[0][1]
            for j in range(table.rows):
                for sm in sim_metas[j].values():
                    sm.matches[i] = [t for t in sm.matches[i] if t[0] in supported]
                    best = max(sm.matches[i], key=lambda t: t[3], default=("", "", None, 0.0))
                    sm.pids[i], sm.labels[i], sm.qids[i], sm.bests[i] = best
        den = max(den, 1)

        for j in range(table.rows):
            for cand in table[k, j]:
                sm = sim_metas[j][cand.qid]
                sm.avg = sum(sm.bests) / den
                for i in range(table.cols):
                    sm.matches[i].sort(key=itemgetter(-1), reverse=True)
                cand.metadata = sm
            # if len(choices := [ca for ca in table[k, j].candidates if ca.metadata.avg > 0]) > 3:
            #     table[k, j].candidates = choices

    def process_rows(self, table: Table, infer_subcol: bool) -> None:
        """对整表做处理"""
        for col in table.columns:
            col.empty = not any(col.cell_texts)
        # 对每列做处理，求评分
        for k in range(table.cols):
            if not (table[k].searchable and (k == table.key_col or infer_subcol)):
                continue
            self.process_rows_sub(table, k)
            table[k].metadata = {
                "major": (sum(max((ca.metadata.avg for ca in cell), default=0) ** 2 for cell in table[k]) / table.rows)
                ** 0.5
            }

        # 然后是找主题列
        if not infer_subcol:
            return
        old_key = table.key_col
        table.key_col = max(range(table.cols), key=lambda k: table[k].metadata["major"] if table[k].searchable else -1)
        if table.sub_col.metadata["major"] < self.key_threshold:
            table.key_col = old_key
            return
        major_threshold = min(self.key_threshold, table.sub_col.metadata["major"] - 0.1)
        table.sub_col.keyable = True
        for i, col in enumerate(table):
            if col.searchable and i != table.key_col and col.metadata["major"] > major_threshold:
                col.keyable = True

    def process(self, table: Table, infer_subcol: bool = False) -> bool:
        # TODO Add searchable check
        self.types = set[str]()
        self.properties = make_list2(set[str], table.cols, table.cols)
        self.entities = make_list2(set[str], table.cols, table.rows)
        self.match_dict = make_list3(dict[tuple[str, str], float], table.cols, table.cols, table.rows)
        self.process_rows(table, infer_subcol=infer_subcol)
        return True

    @classmethod
    @property
    def params(cls):
        return {
            "match_cutoff": cls.match_cutoff,
            "key_threshold": cls.key_threshold,
            "property_select_threshold": cls.property_select_threshold,
        }
