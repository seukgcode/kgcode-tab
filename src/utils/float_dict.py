import operator
from collections import defaultdict
from typing import Iterable, Self


class FloatDict(defaultdict[str, float]):
    def __init__(self, *args) -> None:
        super().__init__(float, *args)

    def __iadd__(self, other: Self) -> Self:
        for k, v in other.items():
            self[k] += v
        return self

    def __add__(self, other: Self) -> Self:
        ret = FloatDict(self)
        ret += other
        return ret

    def __imul__(self, other: float) -> Self:
        for k, v in self.items():
            self[k] *= other
        return self

    def __mul__(self, other: float) -> Self:
        return FloatDict({k: v * other for k, v in self.items()})

    def __itruediv__(self, other: float) -> Self:
        for k, v in self.items():
            self[k] /= other
        return self

    def __truediv__(self, other: float) -> Self:
        return FloatDict({k: v / other for k, v in self.items()})

    def set_max(self, k: str, v: float) -> None:
        self[k] = max(self[k], v)

    def topmost(self, top_num: int) -> list[tuple[str, float]]:
        return sorted(self.items(), key=operator.itemgetter(1), reverse=True)[:top_num]

    def top_one(self, *, cutoff: float = 0) -> str | None:
        if len(self) == 0:
            return None
        top = max(self.items(), key=operator.itemgetter(1))
        return top[0] if top[1] >= cutoff else None

    def over(self, *, cutoff: float = 0) -> "FloatDict":
        """Get a subset where value is over cutoff

        Args:
            cutoff (float, optional): Score cutoff. Defaults to 0.

        Returns:
            FloatDict: A new FloatDict.
        """
        res = FloatDict()
        for k, v in self.items():
            if v >= cutoff:
                res[k] = v
        return res

    def any_greater(self, score: float = 0) -> bool:
        return any(v > score for v in self.values())

    def sum(self) -> float:
        return sum(self.values())

    def max(self) -> float:
        return max(self.values(), default=0)

    def maximums(self, epsilon: float = 1e-6) -> Iterable[str]:
        m = max(self.values(), default=0)
        return (k for k, v in self.items() if v > m - epsilon)

    def normalize_avg(self) -> None:
        s = self.sum()
        if s > 0:
            for k in self.keys():
                self[k] /= s

    def normalize(self) -> Self:
        # s = self.sum()
        # if s > 0:
        #     for k in self.keys():
        #         self[k] /= s

        import numpy as np

        if len(self) == 0:
            return self
        a = np.array(list(self.values()))
        d = np.max(a)
        den = np.sum(np.exp(a - d))
        for k, v in self.items():
            self[k] = np.exp(v - d) / den
        return self

    def __repr__(self) -> str:
        return super(dict).__repr__()

    def __str__(self) -> str:
        return super(dict).__str__()
