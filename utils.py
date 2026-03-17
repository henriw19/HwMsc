from typing import Iterable, Any, List, Dict


def flatten(xss: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for xs in xss for x in xs]

def flatten_dicts(kvss: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    result = {}
    for kvs in kvss:
        for k, v in kvs.items():
            if k in result:
                raise ValueError(
                    "Couldn't flatten dictionaries with key conflicts. " +
                    f"Conflicting key: {k}. " +
                    f"Dictionaries: {kvss}")
            else:
                result[k] = v
    return result