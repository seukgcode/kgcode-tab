from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Callable, Iterable, TypeAlias

from ...process.filters import order_by_rank
from ...table.table_data import Candidate, Cell, Column
from ..utils import max_many, max_many_eps

if TYPE_CHECKING:
    from .annotator import TableAnnotator


@dataclass
class FilterContext:
    a: TableAnnotator
    i: int
    j: int
    col: Column
    cell: Cell


CandidateFilter: TypeAlias = Callable[[Iterable[Candidate] | None, FilterContext], Iterable[Candidate] | None]


def pipe(
    filters: Iterable[CandidateFilter], candidates: Iterable[Candidate], ctx: FilterContext
) -> list[Candidate] | None:
    red = reduce(lambda t, s: s(t, ctx), filters, candidates)
    return list(red) if red is not None else None


# 目前规则是，None表示结果留空

# def primary_maximums(threshold: float):
#     def wrapped(candidates: Iterable[Candidate] | None, ctx: FilterContext):
#         if candidates is None: return None
#         m = ctx.a.cea_rating[ctx.i][ctx.j].maximums(threshold)
#         return filter(lambda ca: ca.qid in m, candidates)

#     return wrapped

CandidateSelector: TypeAlias = Callable[[FilterContext], Candidate | str | None]


def default_select(
    *,
    eq_skip: bool = False,
    fallback_rank: bool = True,
    primary_eps: float = 1e-3,
    tertiary_eps: float = 1e-2,
    ratio_eps: float = 1e-2,
):
    def impl(ctx: FilterContext):
        from rapidfuzz import fuzz
        from rapidfuzz.utils import default_process

        i, j = ctx.i, ctx.j
        answer_many = set(ctx.a.cea_rating[i][j].maximums(primary_eps))
        ctx.a.assign_cea_ranking(i, j, list(answer_many))
        if len(answer_many) == 1:
            return next(iter(answer_many))

        # 选answer_many里编辑距离最小，然后rank最小
        if choices := [ca for ca in ctx.cell.candidates if ca.qid in answer_many]:
            ctx.a.assign_cea_ranking(i, j, sorted(choices, key=lambda c: (-c.score, c.rank)))
            if eq_skip and len(answer_many) == len(ctx.a.cea_rating[i][j]) == len(choices):
                return None

            bests = choices
            bests = ctx.a.choose_from_props_many(i, bests, eps=tertiary_eps)
            ctx.a.assign_cea_ranking(i, j, sorted(bests, key=lambda c: (-c.score, c.rank)))

            bests = max_many_eps(
                bests,
                key=lambda c: fuzz.WRatio(ctx.cell.value, c.entity.label, processor=default_process),
                eps=ratio_eps,
            )
            ctx.a.assign_cea_ranking(i, j, sorted(bests, key=lambda c: c.rank))
            best = min(bests, key=lambda c: c.rank)
            # if len(bests) > 1:
            #     self.cea[i][j] = self.all_cea_scores["semantic_col"][i][j].top_one()
            # else:
            #     self.cea[i][j] = bests[0].qid
            # continue
            return best.qid
        # self.cea[i][j] = self.choose_from_props(i, self.table[i, j], answer_many)
        # 下面是仅仅选 rank 最小的
        if fallback_rank:
            ranks = {ca.qid: ca.rank for ca in ctx.cell.candidates}
            ctx.a.assign_cea_ranking(i, j, sorted(answer_many, key=lambda x: ranks.get(x, 1)))
            return min(answer_many, key=lambda x: ranks.get(x, 1))
        return None

    return impl


# def rank_guard():
#     def wrapped(candidates: Iterable[Candidate]| None, ctx: FilterContext):
#         if candidates is None:
#             ranks = {ca.qid: ca.rank for ca in ctx.cell.candidates}
#                 self.cea[i][j] = min(answer_many, key=lambda x: ranks.get(x, 1))
#     return wrapped
