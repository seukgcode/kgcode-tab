from typing import Any, Callable, Iterable, Type, TypeVar

T = TypeVar('T')


def make_list1(tp: Type[T], len0: int) -> list[T]:
    return [tp() for _ in range(len0)]


def make_list2(tp: Type[T], len0: int, len1: int) -> list[list[T]]:
    return [[tp() for _ in range(len1)] for _ in range(len0)]


def make_list3(tp: Type[T], len0: int, len1: int, len2: int) -> list[list[list[T]]]:
    return [[[tp() for _ in range(len2)] for _ in range(len1)] for _ in range(len0)]


def max_many(l: Iterable[T], key: Callable[[T], Any]) -> list[T]:
    max_value = key(max(l, key=key))
    return [x for x in l if key(x) == max_value]
