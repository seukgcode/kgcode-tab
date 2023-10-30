import heapq
from pathlib import Path
from typing import Any, Iterable

import torch

from ..process.abstract import make_abstract_from_given_properties
from ..process.st_model import encode_for_emb
from ..table.entity import Entity
from ..utils.float_dict import FloatDict
from .utils import make_list1


class EmbeddingScorer:
    # embeddings: dict[str, torch.Tensor]
    def __init__(self, cache_path: Any = ".cache/embedding.pkl") -> None:
        pth = Path(cache_path)
        # self.embeddings = torch.load(pth, "cuda") if pth.exists() else {}

    @staticmethod
    def score(
        candidates: list[list[Entity]], primary_scores: list[FloatDict], given: set[str], K: int
    ) -> list[FloatDict]:
        """对一列实体，给新的评分

        Args:
            candidates (list[list[Candidate]]): 这列的候选实体
            primary_scores (list[FloatDict]): 基础分数
            given (set[str]): 用于生成摘要的给定候选属性集
            K (int): 近邻数

        Returns:
            list[FloatDict]: 这列的新评分
        """
        entities = list({c.qid: c for cl in candidates for c in cl}.values())
        if not entities:
            return make_list1(FloatDict, len(candidates))
        # print("num", len(entities), len(given))
        abstracts = make_abstract_from_given_properties(entities, given)
        emb = encode_for_emb(abstracts)
        emb_dict = {e.qid: em for e, em in zip(entities, emb)}

        def get_distance(q: str, tl: list[tuple[str, int]]) -> Iterable[tuple[float, float]]:
            if not tl:
                return []
            dist = torch.pairwise_distance(emb_dict[q].repeat(len(tl), 1), torch.stack([emb_dict[t[0]] for t in tl]))
            return zip((primary_scores[t[1]][t[0]] for t in tl), dist.cpu().numpy())

        new_scores: list[FloatDict] = [
            FloatDict(
                {
                    c.qid: sum(
                        (
                            s / d
                            for s, d in heapq.nsmallest(
                                K,
                                get_distance(
                                    c.qid, [(c2.qid, j) for j, cl2 in enumerate(candidates) if j != i for c2 in cl2]
                                ),
                                key=lambda t: t[1],
                            )
                        )
                    )
                    / K
                    if len(cl) > 1
                    else 1
                    for c in cl
                }
            )
            for i, cl in enumerate(candidates)
        ]
        # print(emb)
        # pprint(new_scores, width=120)
        return new_scores
