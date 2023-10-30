from concurrent.futures import ThreadPoolExecutor, wait
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar, Iterable

import orjson as json
import ormsgpack

from ..searchmanage import SearchManage
from ..table import Entity
from ..utils import PathLike, config, logger
from . import dbhelper
from .chunking import search_chunk

logger.info("Load wikibase")


class EntityManager:
    sc: ClassVar[SearchManage] = SearchManage(key="ids", m_num=config.get_("process.entity.concurrency", 200))
    entities: ClassVar[dict[str, Entity]] = {}

    search_keys: ClassVar[list[str]] = ["labels/en", "descriptions/en", "aliases/en"]
    search_keys_p: ClassVar[list[str]] = ["labels/en", "descriptions/en", "aliases/en", "properties"]

    @classmethod
    def store_wikidata_entities(
        cls, qids: Iterable[str], with_properties: bool = True, *, force: bool = False, level: int = 0
    ) -> None:
        qids = set(qids)
        logger.info("Start detail searching (with_propertities=%s, level=%s). ", with_properties, level)

        # 减去数据库里有的
        qids_search = qids if force else dbhelper.get_wd_entity_keys(qids, with_properties)
        logger.info("Need to search %d/%d entities in total.", len(qids_search), len(qids))

        if qids_search:
            chunk_size = config.get_("process.entity.chunk_size", 2000)
            search_chunk(list(qids_search), chunk_size, cls.chunked_search, with_properties=with_properties)

        if with_properties:
            # 找properties里的id
            ids = dbhelper.get_wd_ids_in_properties(qids) - qids
            if level > 0:
                ids2 = dbhelper.get_wd_ids_of_P31(qids) - qids
                logger.info("Need to search %d+%d entities from properties (recursive).", len(ids), len(ids2))
                cls.store_wikidata_entities(ids - ids2, False)
                cls.store_wikidata_entities(ids2, True, level=level - 1)
            else:
                logger.info("Need to search %d entities from properties.", len(ids))
                cls.store_wikidata_entities(ids, False)

        logger.info("Detail searching completed.")

    @classmethod
    def chunked_search(cls, qids_search: list[str], with_properties: bool = True) -> None:
        res = cls.sc.search_run(
            qids_search,
            timeout=config.get_("process.entity.timeout", 30),
            block_num=config.get_("process.entity.block_num", 3),
            keys=cls.search_keys_p if with_properties else cls.search_keys
        )

        dbhelper.add_wd_entities(qids_search, res)

    @classmethod
    def store_wikidata_entity(cls, qid: str):
        cls.chunked_search([qid])

    @classmethod
    def has(cls, qid: str) -> bool:
        return qid in cls.entities

    @classmethod
    def get(cls, qid: str, force: bool = False) -> Entity:
        e = cls.entities.get(qid, None)
        if not e or e.label is None or force and e.properties is None:
            ee = dbhelper.get_wd_entity(qid)
            if not ee or ee[1] is None or force and ee[4] is None:
                logger.debug(f"Get {qid} as need.")
                cls.store_wikidata_entity(qid)
                ee = dbhelper.get_wd_entity(qid)
            e = Entity(
                qid=ee[0],
                label=ee[1],
                description=ee[2],
                aliases=json.loads(ee[3]),
                properties=json.loads(ee[4]) if ee[4] is not None else None
            )
            cls.entities[qid] = e
        return e

    @classmethod
    def gets(cls, qids: list[str], force: bool = False) -> list[Entity]:
        return [cls.get(q, force) for q in qids]

    @classmethod
    def get_type_ancestors(cls, entity: Entity, depth: int = 3) -> Iterable[tuple[int, str]]:
        assert entity.properties
        classes: list[str] = [p[1] for p in entity.properties.get("P31", [])]
        yield from ((0, x) for x in classes)
        for i in range(1, depth):
            classes = list(chain.from_iterable(EntityManager.get_subclasses(c) for c in classes))
            yield from ((i, x) for x in classes)

    @classmethod
    def get_subclasses(cls, qid: str) -> Iterable[str]:
        """entity是一个 instance of 的实体"""
        return cls.get(qid, force=True).iter_subclasses()

    @classmethod
    def __load_mt(cls, path: Path) -> dict[str, Any]:
        def run(p: Path) -> dict:
            return {k: Entity.from_dict(v) for k, v in ormsgpack.unpackb(p.read_bytes()).items()}

        from itertools import chain
        with ThreadPoolExecutor(16) as executor:
            tasks = []
            for i in range(100):
                if not (p := path.with_suffix(f".part{i}{path.suffix}")).exists():
                    break
                tasks.append(executor.submit(run, p))
            return dict(chain.from_iterable(t.result().items() for t in tasks))

    @classmethod
    def load(cls, path_: PathLike) -> None:
        path = Path(path_)
        if not path.exists():
            return
        logger.info("Load entities from %s.", path_)
        if path.with_suffix(f".part0{path.suffix}").exists():
            logger.info("Loading chunks with multithread.")
            data = cls.__load_mt(path)
        else:
            data = ormsgpack.unpackb(path.read_bytes())
            cls.entities.update({k: Entity.from_dict(v) for k, v in data.items()})
        logger.info("Loading completed.")

    @classmethod
    def __dump_mt(cls, path: Path) -> None:
        def split_list_by_n(list_collection, n):
            for i in range(0, len(list_collection), n):
                yield list_collection[i : i + n]

        def run(keys: list[str], p: Path):
            with open(p, "wb") as f:
                f.write(ormsgpack.packb({k: cls.entities[k] for k in keys}))

        with ThreadPoolExecutor(16) as executor:
            tasks = []
            for i, keys in enumerate(split_list_by_n(list(cls.entities.keys()), 100000)):
                tasks.append(executor.submit(run, keys, path.with_suffix(f".part{i}{path.suffix}")))
            wait(tasks)

    @classmethod
    def dump(cls, path: PathLike) -> None:
        # data = {k: v.to_dict() for k, v in cls.entities.items()}
        with open(path, "wb") as f:
            # msgpack.dump(data, path)
            logger.info("Dump entities to %s.", path)
            f.write(ormsgpack.packb(cls.entities))
        logger.info("Dumping completed.")
