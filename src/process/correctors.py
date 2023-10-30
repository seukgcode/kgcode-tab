import time
from abc import abstractmethod
from itertools import product
from operator import itemgetter
from typing import ClassVar, Iterable, Literal, LiteralString

from rapidfuzz import fuzz, process
from typing_extensions import override

from ..searchmanage.SearchManage import SpellCheck, Wikipedia
from ..searchmanage.tools import AnalysisTools
from ..utils import config, logger
from ..utils.lists import unique
from ..utils.strings import despaces, remove_symbols, split_word
from . import dbhelper

logger.info("Load correctors")


class CheckRequester:
    def __init__(self) -> None:
        ...

    @abstractmethod
    def __call__(self, values: list[str]) -> list[list[str]]:
        ...


class BingRequester(CheckRequester):
    def __init__(self, *, concurrency: int = 50, timeout: float = 30, block_num: int = 2) -> None:
        super().__init__()
        self.timeout = timeout
        self.block_num = block_num
        self.sp = SpellCheck(m_num=concurrency)

    @override
    def __call__(self, values: list[str]) -> list[list[str]]:
        return self.sp.search_run(
            values, timeout=self.timeout, block_num=self.block_num, function_=AnalysisTools.bing_page
        )


class WikipediaRequester(CheckRequester):
    def __init__(self, *, concurrency: int = 50, timeout: float = 30, block_num: int = 2) -> None:
        super().__init__()
        self.timeout = timeout
        self.block_num = block_num
        self.wp = Wikipedia(m_num=concurrency)

    @override
    def __call__(self, values: list[str]) -> list[list[str]]:
        res: list[str] = self.wp.search_run(values, timeout=self.timeout, block_num=self.block_num)
        return [[y or x] for x, y in zip(values, res)]


class SpellChecker:
    check_repeat: ClassVar[int] = config.get_("process.spellcheck.check_repeat", 10)
    chunk_size: ClassVar[int] = config.get_("process.spellcheck.chunk_size", 1000)
    cooldown: ClassVar[float] = config.get_("process.spellcheck.cooldown", 10.0)
    combine_cutoff: ClassVar[float] = config.get_("process.spellcheck.combine_cutoff", 90)
    correction_limit: ClassVar[int] = config.get_("process.spellcheck.correction_limit", 5)
    markers: ClassVar[list[LiteralString]] = [" - Wikipedia", " - Wikidata", " - Wikimedia Commons"]

    @classmethod
    def spell_check_process(
        cls,
        texts: Iterable[str],
        checkers: list[CheckRequester],
        *,
        force: bool | Literal["add"] = False,
    ) -> set[str]:
        texts = {s for s in texts if s}  # 去除空白
        if force == True:
            dbhelper.remove_correction_keys(texts)
        to_correct = texts if force else dbhelper.get_correction_keys(texts)
        logger.info("Start correcting %d/%d texts.", len(to_correct), len(texts))
        if not to_correct:
            return cls.get_all(texts)

        cls.repeated_search(to_correct, checkers)

        all_results = cls.get_all(texts)
        logger.info("Correction completed. Found %d corrections in total.", len(all_results))
        return all_results

    @classmethod
    def repeated_search(cls, to_correct: set[str], checkers: list[CheckRequester]) -> None:
        # 现在工作流是，对全体搜，没搜到的进入下一轮  搜到的就处理
        residuals = [(s, split_word(remove_symbols(s))) for s in to_correct]

        def chunked(values: list[tuple[str, str]], chunk_size: int):
            for j in range(0, len(values), chunk_size):
                logger.info(
                    "  - Spell check request chunk %d/%d of size %d.",
                    j // chunk_size + 1,
                    (len(values) + chunk_size - 1) // chunk_size,
                    chunk_size,
                )
                yield cls._chunked_search(values[j : j + chunk_size], checkers)

        for i in range(cls.check_repeat):
            logger.info("= Spell check request round %d/%d:", i + 1, cls.check_repeat)
            residuals = sum(chunked(residuals, cls.chunk_size), [])
            logger.info("Residuals: %d/%d", len(residuals), len(to_correct))

            if not residuals:  # 如果全纠错完成就结束
                break

            if i == cls.check_repeat // 2:
                logger.warning("Request too many times. Waiting for %ds.", cls.cooldown)
                time.sleep(cls.cooldown)

        if residuals:
            logger.warning("Spell check error (%d): %s", len(residuals), residuals)
            dbhelper.add_correction_failures(map(itemgetter(0), residuals))
            for r in residuals:
                cls._process_result(*r, [])

    @classmethod
    def _chunked_search(
        cls, to_correct: list[tuple[str, str]], checkers: list[CheckRequester]
    ) -> list[tuple[str, str]]:
        reqs = [chk([a[1] for a in to_correct]) for chk in checkers]  # 所有结果
        temp = [sum(x, []) for x in zip(*reqs)]

        for a, b in zip(to_correct, temp):
            if b:
                cls._process_result(*a, b)

        return [a for a, b in zip(to_correct, temp) if not b]  # returns residuals

    @classmethod
    def _process_result(cls, text_orig: str, texts_pre: str, re_check: list[str]) -> None:
        # 如果没成功找到，加到新的列表里去
        if not re_check:
            dbhelper.add_correction(text_orig, unique([texts_pre.lower(), text_orig.lower()]))
            return

        re_check = unique(re_check)
        # res_ = [despaces(bing.replace(m, "")).lower() \
        #         for m in cls.markers for bing in re_check if m in bing] # 去除空格，统一小写
        # res_ = remove_false(res_)
        res_ = re_check

        # 把所有查询结果join起来分词，无参split可以不返回空格
        all_tokens = unique(" ".join(despaces(s.lower()) for s in re_check).split())
        query_tokens = texts_pre.lower().split()  # 对原实体分词

        # 下面原本用lev distance直接做，现在改成fuzz
        # 对entity每个词，找匹配度最高的前两个
        a: list[list[str]] = [
            [t[0] for t in process.extract(m, all_tokens, scorer=fuzz.ratio, score_cutoff=cls.combine_cutoff, limit=2)]
            for m in query_tokens
        ]
        prod = [despaces(" ".join(x)) for x in product(*a)]  # 对每个位置上匹配的进行排列组合……

        choices = list({*res_, *prod, text_orig.lower(), texts_pre.lower()})

        final_res = [
            t[0]
            for t in process.extract(
                text_orig.lower(), choices, score_cutoff=cls.combine_cutoff, limit=cls.correction_limit
            )
        ]

        dbhelper.add_correction(text_orig, final_res)

    @classmethod
    def get_all(cls, texts: Iterable[str]) -> set[str]:
        return set(dbhelper.get_corrections(texts))

    @classmethod
    def get(cls, text: str) -> list[str]:
        return list(dbhelper.get_correction(text))
