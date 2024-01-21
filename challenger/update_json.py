import collections.abc
import json


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



j1 = json.load(open("./resultt.json", "r"))
j2 = json.load(open("./result.json", "r"))
json.dump(update(j1, j2), open("./resultt.json", "w"))
