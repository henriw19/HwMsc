from typing import Iterable, Any, List


def flatten(xss: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for xs in xss for x in xs]