from functools import reduce
from typing import Any, Callable, Iterable, TypeAlias, cast

from more_itertools import take
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from ..table.table_data import Candidate, Cell

CandidateFilter: TypeAlias = Callable[[Iterable[Candidate]], Iterable[Candidate]]


def pipe(filters: Iterable[CandidateFilter], candidates: Iterable[Candidate]) -> list[Candidate]:
    return list(reduce(lambda t, s: s(t), filters, candidates))


def limiter(limit: int):
    def wrapped(candidates: Iterable[Candidate]):
        return take(limit, candidates)

    return wrapped


def order_by_rank(candidates: Iterable[Candidate]):
    return sorted(candidates, key=lambda ca: ca.rank)


def order_by(key: Callable[[Candidate], Any], reverse: bool = False):
    def wrapped(candidates: Iterable[Candidate]):
        return sorted(candidates, key=key, reverse=reverse)

    return wrapped


def filter_ambiguity(candidates: Iterable[Candidate]):
    return filter(
        lambda c: (
            c.entity.label and c.entity.label.lower() != "wikimedia disambiguation page" and
            (c.entity.description or "").lower() != "wikimedia disambiguation page"
        ), candidates
    )


def scorer(fn: Callable[[Candidate], float]):
    def wrapped(candidates: Iterable[Candidate]):
        def _tap(ca: Candidate):
            ca.score = fn(ca)
            return ca

        return map(_tap, candidates)

    return wrapped


def score_by_ratio(ratio=fuzz.ratio):
    return scorer(
        lambda c: cast(
            float,
            max((ratio(cast(Cell, c.cell).text, a, processor=default_process) for a in c.entity.iter_aliases()),
                default=0.)
        ) / 100
    )
