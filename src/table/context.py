from enum import Enum
from typing import ClassVar


class KG(Enum):
    WikiData = "wd"
    DBPedia = "dbp"


class KGContext:
    context_stack: ClassVar[list[KG]] = []

    def __init__(self, kg: KG) -> None:
        self.kg = kg

    def __enter__(self):
        self.context_stack.append(self.kg)

    def __exit__(self, exc_type, exc_value, traceback):
        self.context_stack.pop()
