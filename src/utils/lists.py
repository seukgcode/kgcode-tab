from typing import Any, Iterable, Type, TypeAlias, TypeGuard, TypeVar, overload

T = TypeVar("T", int, float, bool, str)
U = TypeVar("U")
NestedList: TypeAlias = list["NestedList[T]" | T]
BasicType: TypeAlias = int | float | bool | str
MaybeNested: TypeAlias = T | NestedList[T]


def is_list_of(li: list[Any], tp: Type[U]) -> TypeGuard[list[U]]:
    assert li
    return isinstance(li[0], tp)


@overload
def super_flat(list_like: T, flatted: list[T]) -> int:
    ...


@overload
def super_flat(list_like: NestedList[T], flatted: list[T]) -> NestedList[int]:
    ...


def super_flat(list_like: T | NestedList[T], flatted: list[T]) -> int | NestedList[int]:
    if isinstance(list_like, list):
        return [super_flat(a, flatted) for a in list_like]
    # if list_like in flatted:
    #     return flatted.index(list_like)
    flatted.append(list_like)
    return len(flatted) - 1


def super_flat_back(indices: int | NestedList[int], flatted: list[T]) -> T | NestedList[T]:
    if isinstance(indices, list):
        return [super_flat_back(i, flatted) for i in indices]
    return flatted[indices]


A = TypeVar("A")


def make_list(a: A | list[A]) -> list[A]:
    return a if isinstance(a, list) else [a]


def remove_false(a: Iterable[A]) -> list[A]:
    return [s for s in a if s]


def unique(a: Iterable[A]) -> list[A]:
    return list(set(a))


def unique_by(a: Iterable[dict[str, A]], *, key: str) -> list[dict[str, A]]:
    return list({x[key]: x for x in a}.values())


def transpose_dictlist(obj: dict[str, list[A]]) -> list[dict[str, A]]:
    keys = obj.keys()
    return [dict(zip(keys, x)) for x in zip(*obj.values())]


def transpose_dictlist_2d(obj: dict[str, list[list[A]]]) -> list[list[dict[str, A]]]:
    return [transpose_dictlist(x) for x in transpose_dictlist(obj)]


def transpose_dictlist_3d(obj: dict[str, list[list[list[A]]]]) -> list[list[list[dict[str, A]]]]:
    return [transpose_dictlist_2d(x) for x in transpose_dictlist(obj)]


def transpose_dictlist_4d(obj: dict[str, list[list[list[list[A]]]]]) -> list[list[list[list[dict[str, A]]]]]:
    return [transpose_dictlist_3d(x) for x in transpose_dictlist(obj)]


if __name__ == "__main__":
    # original = ["5", '1', ["66", ["8", "5", "9"], '2', "77", [['1', '2']]]]
    original: NestedList[int] = [[[1, 2, 3], [4, 5, 6], [7, 2, 3]]]
    # original = [5, 6]
    print(original)
    flatted = []
    print(indices := super_flat(original, flatted))
    print(flatted)
    dedup = {x: x * 2 for x in flatted}
    print(dedup)
    flatted = [dedup[x] for x in flatted]
    print(flatted)
    print(super_flat_back(indices, flatted))

    dl = {"a": [[1, 2], [3, 4]], "b": [[7, 8], [9, 10]]}
    print([transpose_dictlist(x) for x in transpose_dictlist(dl)])
