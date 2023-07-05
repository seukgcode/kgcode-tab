import numpy as np
import pandas as pd

from ...table.table_data import Candidate

from ...table import Table, Column, Cell
from ...utils import FloatDict
from ..utils import make_list1, make_list2, max_many
from .valuematch import SimilarityMeta, ValueMatchWikidata


class TableAnnotator:
    '''
    Core class to process CTA, CEA and CPA for one table.
    '''

    normal_factor: float = 16  # 这个参数跟数据集有关
    fallback_threshold: float = 0.01
    type_ancestors: bool = False
    ancestor_rate: float = 0.8
    ancestor_level: int = 3
    infer_subcol: bool = True
    multi_subcol: bool = False

    def __init__(self, table: Table) -> None:
        '''
        Args:
            data_or_path (Table | str | Path): Instance of Table, or path of json file.
        '''
        self.table = table
        rows, cols = self.table.shape
        self.table.key_col = self.table.key_col
        self.cta_rating = make_list1(FloatDict, cols)
        self.cpa_rating = make_list2(FloatDict, cols, cols)
        self.cea_rating = make_list2(FloatDict, cols, rows)
        self.cta_fallback = make_list1(bool, cols)
        self.cea_fallback = make_list2(bool, cols, rows)
        self.prop_rating = make_list1(FloatDict, cols)

    def init_all(self):
        vm = self.vm
        vm.process(self.table, self.infer_subcol)

        self.types = vm.types

    def sim_norm(self, x: float) -> float:
        return x**self.normal_factor

    def coop(self, sim: float, other: float) -> float:
        return self.sim_norm(sim) * other

    def should_fallback(self, rating: FloatDict, is_col: bool = False) -> bool:
        return rating.max() < self.fallback_threshold * (self.table.rows if is_col else 1)

    def sum_type_score(self, col: Column) -> FloatDict:
        rating0 = FloatDict()
        for i in range(self.table.rows):
            rating2 = FloatDict()
            for c in col[i]:
                for t in c.iter_typeid():
                    rating2[t] = 0.1
            rating0 += rating2
        return rating0

    def cta_from_types(self, col: Column):
        rating0 = FloatDict()
        for i in range(self.table.rows):
            rating2 = FloatDict()
            for c in col[i]:
                if self.type_ancestors:
                    for z, t in c.iter_type_ancestors(self.ancestor_level):
                        rating2.set_max(t, c.score * self.ancestor_rate**z)
                else:
                    for t in c.iter_typeid():
                        rating2.set_max(t, c.score)
            rating0 += rating2
        return rating0

    def cea_from_cta(self, cell: Cell, cta_rating: FloatDict) -> FloatDict:
        rating0 = FloatDict()
        for c in cell:
            rating0.set_max(
                c.qid,
                max((cta_rating[t] * c.score
                     for z, t in c.iter_type_ancestors(self.ancestor_level)) if self.type_ancestors else
                    (cta_rating[t] * c.score for t in c.iter_typeid()),
                    default=0)
            )
        return rating0

    def cea_from_props(self, cell: Cell, i: int) -> FloatDict:
        rating0 = FloatDict()
        for c in cell:
            rating0.set_max(c.qid, sum((self.prop_rating[i][ep] * c.score for ep in c.iter_entity_props())))
        return rating0

    def sum_prop(self) -> None:
        for i, col in enumerate(self.table):
            if not col.searchable:
                continue
            rating = self.prop_rating[i]
            for cell in col:
                rating2 = FloatDict()
                for c in cell:
                    # for ep in c.iter_entity_props():
                    #     rating2.set_max(ep, c.score)
                    for k, v in c.iter_entity_prop_groups():
                        if k == "P31": continue
                        for vv in v:
                            rating2.set_max(f"{k}={vv}", c.score)
                            # 如果类型不匹匹配？
                rating += rating2
            a = rating.topmost(10)
            rating.clear()
            rating.update(dict(a))
            # for k, v in rating.items():
            #     if v < 3:
            #         rating[k] = 0
            rating.normalize()
        '''
        匹配度怎么考虑？
        '''

    def CTA_subject(self, i: int) -> None:
        '''
        Process CTA for subject column.
        '''
        # 对主题列，找 AvgSimilarity 最高的
        rating = self.cta_rating[i]
        for cell in self.table[i]:
            rating2 = FloatDict()
            for c in cell:
                if not isinstance(c.metadata, SimilarityMeta): continue
                sim_meta = c.metadata
                if sim_meta.avg < 1e-3: continue
                if self.type_ancestors:
                    for z, t in c.iter_type_ancestors(self.ancestor_level):
                        rating2.set_max(t, self.sim_norm(sim_meta.avg) * self.ancestor_rate**z)
                else:
                    for t in c.iter_typeid():
                        if t not in self.types:
                            continue
                        rating2.set_max(t, self.sim_norm(sim_meta.avg))
            rating += rating2
        if self.should_fallback(rating, True):
            rating.clear()
            rating.update(self.cta_from_types(self.table[i]))
            self.cta_fallback[i] = True
        rating.normalize()

    def CTA_subject_all(self) -> None:
        for i, col in enumerate(self.table):
            if i == self.table.key_col or self.multi_subcol and col.keyable:
                self.CTA_subject(i)

    def CEA_subject(self, i: int) -> None:
        '''
        Process CEA for subject column.
        '''
        # 根据主题列的p31筛选主题列的实体
        subject_rating = self.cta_rating[i]
        for j, cell in enumerate(self.table[i]):
            rating = self.cea_rating[i][j]
            for c in cell:
                if not isinstance(c.metadata, SimilarityMeta): continue
                sim_meta = c.metadata
                sim = sim_meta.avg
                if sim < 1e-3: continue
                rating[c.qid] += self.coop(
                    sim,
                    max((subject_rating[t]
                         for z, t in c.iter_type_ancestors(self.ancestor_level)) if self.type_ancestors else
                        (subject_rating[t] for t in c.iter_typeid()),
                        default=0)
                )
            if self.should_fallback(rating):
                rating.clear()
                rating.update(self.cea_from_cta(cell, subject_rating))
                self.cea_fallback[i][j] = True
            rating.normalize()

    def CEA_subject_all(self) -> None:
        for i, col in enumerate(self.table):
            if i == self.table.key_col or self.multi_subcol and col.keyable:
                self.CEA_subject(i)

    def CPA_subject(self, k: int) -> None:
        '''
        Process CPA for other columns.
        '''
        # key 为关系，对于每个其他列候选实体，用它的候选项匹配该行主题列实体的属性
        subject_cea_rating = self.cea_rating[k]
        for i in range(self.table.cols):
            if i == k:
                continue
            # 对主题列，找 AvgSimilarity 最高的
            rating = self.cpa_rating[k][i]  # key 为列类的 QID，val 为总共相似评分
            for j in range(self.table.rows):
                rating2 = FloatDict()
                for c in self.table[k, j]:
                    for p in self.vm.properties[k][i]:
                        sim = self.vm.match_dict[k][i][j].get((c.qid, p), 0.0)
                        if sim < 1e-3: continue
                        scea = subject_cea_rating[j][c.qid]
                        rating2.set_max(p, self.coop(sim, scea))
                rating += rating2
            rating.normalize()

    def CPA_all(self):
        for k in range(self.table.cols):
            if self.table[k].searchable:
                self.CPA_subject(k)

    def CTA_other(self, i: int) -> None:
        '''
        Process CTA for other columns.
        '''
        col = self.table[i]
        # 对主题列，找 AvgSimilarity 最高的
        rating = self.cta_rating[i]
        for j, cell in enumerate(col):
            qids = [c.qid for c in cell]  # All candidate qid's of this cell
            rating2 = FloatDict()
            for c in self.table.sub_col[j]:
                if not isinstance(c.metadata, SimilarityMeta): continue
                for p, l, e, s in c.metadata.matches[i]:
                    if e and e in qids:
                        k = qids.index(e)
                        if self.type_ancestors:
                            for z, t in cell[k].iter_type_ancestors(self.ancestor_level):
                                rating2.set_max(t, self.sim_norm(s) * self.ancestor_rate**z)
                        else:
                            for t in cell[k].iter_typeid():
                                rating2.set_max(t, self.sim_norm(s))
            rating += rating2
        if self.should_fallback(rating, True):
            rating.clear()
            rating.update(self.cta_from_types(col))
            self.cta_fallback[i] = True
        rating.normalize()

    def CTA_other_all(self) -> None:
        for i, col in enumerate(self.table):
            if col.searchable and i != self.table.key_col and (not self.multi_subcol or not col.keyable):
                self.CTA_other(i)

    def CEA_other(self, i: int):
        '''
        Process CEA for other columns.
        '''
        # 直接用评分筛除的qid，权重为sim*cea
        # TODO: 对没有答案的，采用同列相似实体匹配的机制
        subject_cea_rating = self.cea_rating[self.table.key_col]
        col = self.table[i]
        # 对主题列，找 AvgSimilarity 最高的
        for j, cell in enumerate(col):
            rating = self.cea_rating[i][j]
            for c in self.table.sub_col[j]:
                base_qid = c.qid
                if not isinstance(c.metadata, SimilarityMeta): continue
                for p, l, e, s in c.metadata.matches[i]:
                    if e:
                        scea = subject_cea_rating[j][base_qid]
                        rating[e] += self.coop(s, scea)
            if self.should_fallback(rating):
                rating.clear()
                rating.update(self.cea_from_cta(cell, self.cta_rating[i]))
                self.cea_fallback[i][j] = True
            rating.normalize()

    def CEA_other_all(self) -> None:
        for i, col in enumerate(self.table):
            if col.searchable and i != self.table.key_col and (not self.multi_subcol or not col.keyable):
                self.CEA_other(i)

    def CTA_sub_answer(self, i) -> None:
        if not self.cta[i]: return
        subcol = self.table[i]
        answer_many = list(self.cta_rating[i].maximums())
        if len(answer_many) <= 1:
            return
        if not self.cta_fallback[i]:
            rating = self.cta_from_types(subcol)
            top = self.cta_rating[i][answer_many[0]]
            answer_many = [k for k in answer_many if rating[k] > top - 1e6]
            if len(answer_many) == 1:
                self.cta[i] = answer_many[0]
                return

        answer_grade = np.zeros(len(answer_many))
        answer_num = np.zeros(len(answer_many))

        for j in range(self.table.rows):
            if not subcol[j].candidates:
                continue
            best_sim = max(c.metadata.avg for c in subcol[j] if isinstance(c.metadata, SimilarityMeta))
            for c in subcol[j]:
                if not isinstance(c.metadata, SimilarityMeta): continue
                if c.metadata.avg != best_sim: continue
                cnt = 0
                for qid in c.iter_typeid():
                    if qid in answer_many:
                        cnt += 1
                        answer_grade[answer_many.index(qid)] += cnt
                        answer_num[answer_many.index(qid)] += 1
        self.cta[i] = answer_many[np.argmin(answer_grade / answer_num)]

    def CTA_other_answer(self, i: int) -> None:
        if not (ans := self.cta[i]): return
        best = self.cta_rating[i][ans]  # 最大匹配度
        answer_many = [k for k, v in self.cta_rating[i].items() if v == best]
        if len(answer_many) <= 1: return

        answer_grade = np.zeros(len(answer_many))
        answer_num = np.zeros(len(answer_many))

        for j in range(self.table.rows):
            cnt = 0
            for c in self.table[i, j]:
                for qid in c.iter_typeid():
                    if qid in answer_many:
                        cnt += 1
                        answer_grade[answer_many.index(qid)] += cnt
                        answer_num[answer_many.index(qid)] += 1
        # if any(answer_num == 0):
        # raise Exception("what the fuck?")
        self.cta[i] = answer_many[np.argmin(answer_grade / answer_num)]

    def CTA_answer_all(self) -> None:
        for i, col in enumerate(self.table):
            if col.keyable:
                self.CTA_sub_answer(i)
            else:
                self.CTA_other_answer(i)

    def choose_from_props(self, i: int, bests: list[Candidate]) -> str:
        # x = {c.qid:sum(self.prop_rating[i][ep] * c.score for ep in c.iter_entity_props()) for c in bests}
        return max(
            bests,
            key=lambda c: (sum((self.prop_rating[i][ep] for ep in c.iter_entity_props())), c.score, -c.rank)
        ).qid

    def choose_from_semantics(self, j: int, bests: list[Candidate]) -> str:
        from ...process.st_model import model
        from sentence_transformers import util
        row_embedding = model.encode([self.table.row_texts[j]])
        embeddings = model.encode([ca.entity.label for ca in bests])
        hits = util.semantic_search(row_embedding, embeddings, top_k=1)  # type: ignore
        ans = hits[0][0]["corpus_id"]
        return bests[ans].qid

    def CEA_answer(self) -> None:
        for i in range(self.table.cols):
            # 如果存在唯一确定的，用它们的属性做参照
            # if determined := {
            #     self.cea[i][j]
            #     for j in range(self.table.rows) if len(list(self.cea_rating[i][j].maximums())) == 1
            # }:
            #     self.prop_rating[i].clear()
            #     for c in self.table[i].iter_candidates():
            #         if c.qid in determined:
            #             for k, v in c.iter_entity_prop_groups():
            #                 if k == "P31": continue
            #                 for vv in v:
            #                     self.prop_rating[i][f"{k}={vv}"] += c.score
            # else:
            #     # 否则用所有
            #     self.prop_rating[i].clear()
            #     for j in range(self.table.rows):
            #         tops = set(self.cea_rating[i][j].maximums())
            #         rating = FloatDict()
            #         for c in self.table[i][j]:
            #             if c.qid in tops:
            #                 for k, v in c.iter_entity_prop_groups():
            #                     if k == "P31": continue
            #                     for vv in v:
            #                         rating.set_max(f"{k}={vv}", c.score)
            #         self.prop_rating[i]+=rating
            # 好像这个还不如整体法
            for j in range(self.table.rows):
                if not self.cea[i][j]: continue
                answer_many = set(self.cea_rating[i][j].maximums())
                if len(answer_many) == 1:
                    continue
                # 选answer_many里编辑距离最小，然后rank最小
                if choices := [ca for ca in self.table[i, j].candidates if ca.qid in answer_many]:
                    bests = max_many(choices, key=lambda c: c.score)
                    # self.cea[i][j] = self.choose_from_semantics(i, bests)
                    self.cea[i][j] = self.choose_from_props(i, bests)
                    # self.cea[i][j] = min(bests, key=lambda c: c.rank).qid
                    continue
                # self.cea[i][j] = self.choose_from_props(i, self.table[i, j], answer_many)
                # 下面是仅仅选 rank 最小的
                ranks = {ca.qid: ca.rank for ca in self.table[i, j].candidates}
                self.cea[i][j] = min(answer_many, key=lambda x: ranks.get(x, 1))

    def CPA_answer_subject(self, k: int, i: int) -> None:
        answer_many = set(self.cpa_rating[k][i].maximums())
        if len(answer_many) == 1:
            return
        rating = FloatDict()
        for j in range(self.table.rows):
            rating2 = FloatDict()
            for c in self.table[k, j]:
                for p in self.vm.properties[k][i]:
                    if p not in answer_many: continue
                    sim = self.vm.match_dict[k][i][j].get((c.qid, p), 0.0)
                    if sim < 1e-3: continue
                    rating2.set_max(p, sim * c.score)
            rating += rating2
        self.cpa[k][i] = rating.top_one()

    def CPA_answer(self) -> None:
        for k in range(self.table.cols):
            if not self.table[k].searchable:
                continue
            for i in range(self.table.cols):
                if i == k or not self.cpa[k][i]:
                    continue
                self.CPA_answer_subject(k, i)

    def process(self):
        '''
        Process all task.

        Returns:
            self
        '''
        self.vm = ValueMatchWikidata()
        self.init_all()
        self.sum_prop()
        self.CTA_subject_all()
        self.CEA_subject_all()
        self.CPA_all()
        self.CTA_other_all()
        self.CEA_other_all()
        self.CEA_subject_all()  # 改善很小
        self.cta = [r.top_one() for r in self.cta_rating]
        self.CTA_answer_all()
        self.cpa = [[r.top_one() for r in rr] for rr in self.cpa_rating]
        self.cea = [[r.top_one() for r in rr] for rr in self.cea_rating]
        self.CEA_answer()
        self.CPA_answer()
        return self

    def print_table(self):
        '''
        Print original table formatted by DataFrame.

        Returns:
            self
        '''
        print(f"Table [{self.table.name}]")
        df = pd.DataFrame([[r.value for r in c] for c in self.table])
        print(df, "\n")
        return self

    def print_result(self):
        '''
        Print annotation result of PIDs and QIDs formatted by DataFrame.

        Returns:
            self
        '''
        print(f"Annotation with table [{self.table.name}]")
        df = pd.DataFrame(self.cea).transpose()
        df.columns = pd.MultiIndex.from_arrays([self.cpa, self.cta], names=("CPA", "CTA"))
        print(df, "\n")
        return self

    @classmethod
    @property
    def params(cls):
        return {
            "infer_subcol": cls.infer_subcol, "multi_subcol": cls.multi_subcol, "normal_factor": cls.normal_factor,
            "fallback_threshold": cls.fallback_threshold, "type_ancestors": cls.type_ancestors, "ancestor_level":
            cls.ancestor_level, "ancestor_rate": cls.ancestor_rate
        }
