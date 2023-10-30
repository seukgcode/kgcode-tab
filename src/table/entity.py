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
        yield self.label
        if self.aliases:
            yield from self.aliases

    def iter_subclasses(self) -> Iterable[str]:
        assert self.properties is not None
        if "P279" in self.properties:
            yield from (pp[1] for pp in self.properties["P279"])

    def iter_props(self) -> Iterable[str]:
        props = self.properties
        assert props is not None
        return (f"{k}={p[1]}" for k, v in props.items() if v and v[0][0] == "wikibase-entityid" for p in v)

    def iter_prop_groups(self) -> Iterable[tuple[str, list[str]]]:
        props = self.properties
        assert props is not None
        return ((k, [p[1] for p in v]) for k, v in props.items() if v and v[0][0] == "wikibase-entityid")

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
