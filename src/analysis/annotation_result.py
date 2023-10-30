from dataclasses import dataclass


@dataclass
class AnnotationResult:
    shape: tuple[int, int]
    subcol: int
    cta: list[str | None]
    cea: list[list[str | None]]
    cpa: list[list[str | None]]
    cea_score: dict[str, list[list[dict[str, float]]]]
    cea_ranking: list[list[list[str]]]
    duration: float


@dataclass
class AnnotationAllResult:
    annotations: dict[str, AnnotationResult]
    params: dict[str, float]
