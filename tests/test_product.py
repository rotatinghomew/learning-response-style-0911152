from collections import Counter
import json
from pathlib import Path
import unittest

from learning.domain import overview, validate


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_ordinal_counts_reconcile(self):
        result = overview(self.tables, {})
        self.assertEqual(sum(sum(row["counts"].values()) for row in result["distributions"]), len(self.tables["answers"]))

    def test_missing_answers_are_not_midpoint(self):
        self.tables["answers"] = []
        validate(self.tables)
        self.assertEqual(overview(self.tables, {})["distributions"], [])

    def test_scale_violation_rejected(self):
        self.tables["answers"][0]["answer"] = 6
        with self.assertRaises(ValueError):
            validate(self.tables)
