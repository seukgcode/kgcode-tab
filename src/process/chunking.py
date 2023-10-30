import time
from typing import Any

from ..utils import logger


def search_chunk(all_list: list[str], chunk_size: int, impl: Any, *args, **kwargs):
    for st in range(0, len(all_list), chunk_size):
        logger.info(
            "Search chunk %d/%d of size %d.",
            st // chunk_size + 1,
            (len(all_list) + chunk_size - 1) // chunk_size,
            chunk_size,
        )
        impl(all_list[st : st + chunk_size], *args, **kwargs)
        time.sleep(1)
