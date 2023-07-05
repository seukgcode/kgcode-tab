from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin


def ensure_startswith(s: str | None, t: str) -> str:
    if not s:
        return ""
    return s if s.startswith(t) or s.lower() == "nil" else t + s


class EvaluationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@dataclass
class EvalResult(DataClassJsonMixin):
    correct: float
    annotated: int
    total: int
    main_score: float
    secondary_score: float
    F1: float
    P: float
    R: float

    def __str__(self) -> str:
        return f"Result[{self.correct}/{self.annotated}/{self.total}, F1={self.F1:.4f}, P={self.P:.4f}, R={self.R:.4f}]"

    @classmethod
    def from_score(cls, correct: float, annotated: int, total: int):
        precision = correct / annotated if annotated > 0 else 0.0
        recall = correct / total
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return cls(
            correct=correct,
            annotated=annotated,
            total=total,
            main_score=f1,
            secondary_score=precision,
            F1=f1,
            P=precision,
            R=recall
        )
