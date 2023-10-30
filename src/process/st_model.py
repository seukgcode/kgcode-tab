from typing import cast
from sentence_transformers import SentenceTransformer
from torch import Tensor
from sentence_transformers import util

# model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda", cache_folder="./.models")
# ss_model = SentenceTransformer(".models/trained-semquery-sm", device="cuda")
# print("Load SentenceTransformer for semantic search:", ss_model, ss_model._target_device)

emb_model = SentenceTransformer(".models/trained3", device="cuda")
print("Load SentenceTransformer for cluster embedding:", emb_model, emb_model._target_device)


def encode_for_ss(sentences: str | list[str]) -> Tensor:
    return cast(Tensor, ss_model.encode(sentences, batch_size=512, convert_to_tensor=True, device="cuda"))
    # sm: 512, lg: 128


def encode_for_ss_li(sentences: str | list[str]) -> list[Tensor]:
    return cast(list[Tensor], ss_model.encode(sentences, batch_size=512, convert_to_numpy=False, device="cuda"))


def encode_for_emb(sentences: str | list[str]) -> Tensor:
    return cast(Tensor, emb_model.encode(sentences, batch_size=64, convert_to_tensor=True, device="cuda"))


def semantic_search(query_embeddings: Tensor, corpus_embeddings: Tensor, top_k: int = 10):
    return util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
