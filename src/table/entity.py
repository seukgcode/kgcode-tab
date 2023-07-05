from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Entity:
    qid: str
    label: str
    description: str
    aliases: list[str] | None = field(default=None)
    properties: dict[str, list[Any]] | None = field(default=None)

    def iter_aliases(self) -> Iterable[str]:
        if self.label:
            yield self.label
        if self.aliases:
            yield from self.aliases

    def iter_subclasses(self) -> Iterable[str]:
        assert self.properties is not None
        if "P279" in self.properties:
            yield from (pp[1] for pp in self.properties["P279"])

    def to_dict(self):
        return {
            "qid": self.qid,
            "label": self.label,
            "description": self.description,
            "aliases": self.aliases,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, kvs: dict):
        return cls(
            qid=kvs["qid"],
            label=kvs["label"],
            description=kvs["description"],
            aliases=kvs["aliases"],
            properties=kvs["properties"],
        )