import re

word_pattern = re.compile(r"[A-Z][^A-Z 0-9]+")
non_alnum_pattern = re.compile(r"[^ 0-9A-Za-z]")


def despaces(s: str) -> str:
    """Remove redundant spaces, including continuous spaces and leading/trailing spaces.

    Args:
        s (str): A string.

    Returns:
        str: The string after space removal
    """
    return " ".join(s.split()).strip(" ‘\".;:,").removesuffix("(").removeprefix(")").removeprefix(",")


def remove_symbols(s: str) -> str:
    return non_alnum_pattern.sub(" ", s)


def split_word(text: str) -> str:
    """Split camelCased or PascalCased text into space separated words.

    TODO This function requires optimization.

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    a = word_pattern.split(text)
    b = word_pattern.findall(text)
    l = min(len(a), len(b))
    temp = [b[i] if a[i] == "" else a[i] + b[i] for i in range(l)]
    temp += a[l :] + b[l :]
    return despaces(" ".join(temp))
