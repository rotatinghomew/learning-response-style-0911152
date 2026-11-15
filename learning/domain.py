from collections import Counter, defaultdict
import math

import numpy as np

from learning.codec import numeric, require_fields, unique

PRODUCT = '学习体验追踪'
KEYS = {'people': ['person'], 'items': ['item'], 'answers': ['person', 'wave', 'item'], 'vignettes': ['vignette'], 'vignette_answers': ['person', 'wave', 'vignette']}


def validate(tables):
    if set(tables) != set(KEYS):
        raise ValueError("资料表集合与产品契约不一致")
    for name, fields in KEYS.items():
        require_fields(tables[name], fields)
        unique(tables[name], fields)
    people = {row["person"] for row in tables["people"]}
    items = {row["item"]: row for row in tables["items"]}
    vignettes = {row["vignette"] for row in tables["vignettes"]}
    for row in tables["answers"]:
        if row["person"] not in people or row["item"] not in items:
            raise ValueError("答卷引用不存在")
        item = items[row["item"]]
        if not item["minimum"] <= row["answer"] <= item["maximum"]:
            raise ValueError("答案超出量尺")
    for row in tables["vignette_answers"]:
        if row["person"] not in people or row["vignette"] not in vignettes or row["answer"] not in range(1, 6):
            raise ValueError("情境评价不合法")


def overview(tables, parameters):
    counts = defaultdict(Counter)
    for row in tables["answers"]:
        counts[(row["wave"], row["item"])][row["answer"]] += 1
    return {"distributions": [{"wave": key[0], "item": key[1], "counts": dict(value)}
                              for key, value in sorted(counts.items())], "people": len(tables["people"])}


ANALYSES = {"overview": overview}
