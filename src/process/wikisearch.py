from abc import abstractmethod
from typing import Any, Iterable, TypedDict

from typing_extensions import override

from ..searchmanage import DbpediaLookUp, SearchManage
from ..utils import logger
from ..utils.lists import transpose_dictlist_2d
from . import dbhelper
from .chunking import search_chunk

logger.info("Load wikisearch")


class WDSearchResult(TypedDict):
    id: str
    label: str
    url: str
    description: str


class SearchHelper:
    def __init__(self, kg: str, *, chunk_size: int) -> None:
        self.kg = kg
        self.chunk_size = chunk_size

    @abstractmethod
    def __call__(self, texts_to_search: list[str]) -> list[list[dict[str, Any]]]:
        pass

    def search_store(self, texts: Iterable[str], *, force: bool = False, check: bool = False) -> None:
        """Search Process form text->IRIs using wikimedia API.

        Args:
            texts (list[str]): _description_
            check (bool, optional): Research empty keys. Defaults to True.
        """
        logger.info("Start entity searching.")
        texts = set(texts)
        if force:
            dbhelper.remove_wd_search_keys(texts)
        texts_to_search = texts if force else dbhelper.get_wd_search_keys(texts, check)

        logger.info("Need to search %d/%d entities in total.", len(texts_to_search), len(texts))
        if not texts_to_search:
            return

        search_chunk(list(texts_to_search), self.chunk_size, self.chunked_search)
        logger.info("Search completed.")

    def chunked_search(self, texts_search: list[str]) -> None:
        re2 = self.__call__(texts_search)
        logger.info("Found %s entities.", sum(bool(a) for a in re2))
        dbhelper.add_wd_search(texts_search, re2)

    def get(self, key: str | list[str]) -> Iterable[tuple[str, str, int]]:
        return dbhelper.get_wd_search(key)


class WDSearch(SearchHelper):
    def __init__(
        self, *, concurrency: int = 50, chunk_size: int = 1000, timeout: float = 30, block_num: int = 3, limit: int = 50
    ) -> None:
        super().__init__("wd", chunk_size=chunk_size)
        self.sm = SearchManage(m_num=concurrency)
        self.timeout = timeout
        self.block_num = block_num
        self.limit = limit

    @override
    def __call__(self, texts_to_search: list[str]) -> list[list[dict[str, Any]]]:
        re_ = self.sm.search_run(
            texts_to_search, keys="all", timeout=self.timeout, block_num=self.block_num, limit=self.limit
        )
        re_ = transpose_dictlist_2d(re_)
        re_ = [[b for b in a if not WDSearch.is_ambiguity(b)] for a in re_]
        return re_

    @staticmethod
    def is_ambiguity(a: dict[str, Any]):
        return (a.get("label") or "").lower() == "wikimedia disambiguation page" or (
            a.get("description") or ""
        ).lower() == "wikimedia disambiguation page"


class DBPSearch(SearchHelper):
    def __init__(self, *, concurrency: int = 100, chunk_size: int = 500) -> None:
        super().__init__("dbp", chunk_size=chunk_size)
        self.dl = DbpediaLookUp(m_num=concurrency)

    @override
    def __call__(self, texts_to_search: list[str]) -> list[list[dict[str, Any]]]:
        re_ = self.dl.search_run(texts_to_search, timeout=30, time_stop=30, block_num=3)
        re2 = transpose_dictlist_2d(re_)
        for a in re_:
            for b in a:
                b["label"] = b["label"].replace("<B>", "").replace("</B>", "")
        return re2
        # label, resource, typeName, type
