from dataclasses import dataclass


@dataclass
class AnnotationResult:
    subcol: int
    cta: list[str | None]
    cea: list[list[str | None]]
    cpa: list[list[str | None]]
