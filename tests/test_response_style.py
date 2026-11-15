import copy
import json
from pathlib import Path
import unittest

from learning.codec import digest, encode
from learning.domain import ANALYSES, validate
from learning.response_style import (
    DEFAULTS, response_style_analysis,
)

SMALL = {"bootstrap": 60, "permutations": 60, "folds": 3}


class ResponseStyleTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def run_analysis(self, tables=None, **overrides):
        parameters = dict(SMALL)
        parameters.update(overrides)
        return response_style_analysis(tables if tables is not None else self.tables,
                                       parameters)

    def test_result_serializes_without_nonfinite(self):
        result = self.run_analysis()
        encoded = encode(result)
        self.assertIn(b"response_style/1", encoded)
        self.assertNotIn(b"NaN", encoded)
        self.assertNotIn(b"Infinity", encoded)

    def test_registered_alongside_overview(self):
        self.assertIn("overview", ANALYSES)
        self.assertIn("response_style", ANALYSES)

    def test_does_not_mutate_evidence(self):
        before = digest(self.tables)
        self.run_analysis()
        self.assertEqual(before, digest(self.tables))

    def test_input_order_independent(self):
        first = self.run_analysis()
        shuffled = {name: list(reversed(rows)) for name, rows in self.tables.items()}
        second = self.run_analysis(shuffled)
        self.assertEqual(digest(first), digest(second))

    def test_holdout_and_fit_are_disjoint_and_cover_all(self):
        result = self.run_analysis()
        design = result["design"]
        fit = set(design["fit_persons"])
        holdout = set(design["holdout_persons"])
        self.assertFalse(fit & holdout)
        self.assertEqual(fit | holdout, {row["person"] for row in self.tables["people"]})
        self.assertGreater(len(holdout), 0)
        self.assertGreater(len(fit), 0)

    def test_holdout_records_do_not_affect_fit_estimates(self):
        """改动 holdout 学员的作答不得改变题目位置、信度或结论。"""
        result = self.run_analysis()
        holdout = set(result["design"]["holdout_persons"])
        altered = copy.deepcopy(self.tables)
        for row in altered["answers"]:
            if row["person"] in holdout:
                row["answer"] = 1 if row["answer"] != 1 else 5
        for row in altered["vignette_answers"]:
            if row["person"] in holdout:
                row["answer"] = 1 if row["answer"] != 1 else 5
        validate(altered)
        changed = self.run_analysis(altered)
        self.assertEqual(result["design"]["fit_persons"], changed["design"]["fit_persons"])
        self.assertEqual(digest(result["item_locations"]), digest(changed["item_locations"]))
        self.assertEqual(digest(result["reliability"]), digest(changed["reliability"]))
        self.assertEqual(digest(result["conclusions"]), digest(changed["conclusions"]))
        self.assertEqual(digest(result["comparability"]), digest(changed["comparability"]))

    def test_holdout_check_reports_change_when_altered(self):
        result = self.run_analysis()
        holdout = set(result["design"]["holdout_persons"])
        altered = copy.deepcopy(self.tables)
        for row in altered["answers"]:
            if row["person"] in holdout:
                row["answer"] = 5
        changed = self.run_analysis(altered)
        self.assertNotEqual(digest(result["holdout"]), digest(changed["holdout"]))

    def test_vignettes_are_not_forced_as_anchors(self):
        result = self.run_analysis()
        for trait, block in result["vignettes"].items():
            with self.subTest(trait=trait):
                self.assertIn(block["status"], {"supported", "unsupported"})
                if not block["order_supported"] or not block["convergence_supported"]:
                    self.assertEqual(block["status"], "unsupported")
        selection = result["model_selection"]
        for trait, block in selection.items():
            supported = result["vignettes"][trait]["status"] == "supported"
            self.assertEqual(block["vignette_included"], supported)
            if not supported:
                self.assertNotIn("vignette", block["models"])

    def test_comparability_can_fail(self):
        """当前数据信度不足；必须允许并明确报告不支持比较。"""
        result = self.run_analysis()
        for trait, block in result["comparability"].items():
            with self.subTest(trait=trait):
                self.assertIsInstance(block["comparable"], bool)
                if not block["comparable"]:
                    self.assertTrue(block["failed_constraints"])
                    self.assertEqual(result["conclusions"][trait]["verdict"],
                                     "not_supported")

    def test_failed_reliability_blocks_claims(self):
        result = self.run_analysis()
        for trait, block in result["reliability"].items():
            if block["status"] != "adequate":
                self.assertFalse(result["comparability"][trait]["comparable"])

    def test_every_change_has_uncertainty(self):
        result = self.run_analysis()
        for trait, block in result["changes"].items():
            for entry in block:
                self.assertIn("ci", entry["overall"])
                self.assertIn("n_pairs", entry["overall"])
                for group in entry["groups"]:
                    self.assertIn("ci", group)
                    self.assertIn("level", group)

    def test_change_verdicts_require_comparability(self):
        result = self.run_analysis()
        for trait, verdicts in result["change_verdicts"].items():
            comparable = result["comparability"][trait]["comparable"]
            for verdict in verdicts:
                with self.subTest(trait=trait, level=verdict["level"]):
                    if not comparable:
                        self.assertFalse(verdict["supported"])

    def test_sensitivity_covers_items_groups_and_directions(self):
        result = self.run_analysis()
        for trait, block in result["sensitivity"].items():
            scenarios = [row["scenario"] for row in block["scenarios"]]
            self.assertIn("base:balanced", scenarios)
            self.assertIn("forward_only", scenarios)
            self.assertIn("reverse_only", scenarios)
            items = [row["item"] for row in self.tables["items"]
                     if row["trait"] == trait]
            for item in items:
                self.assertIn(f"exclude_item:{item}", scenarios)
            for field in ("site", "group"):
                for level in sorted({row[field] for row in self.tables["people"]}):
                    self.assertIn(f"exclude_group:{field}={level}", scenarios)
            self.assertIn("depends_on_items", block)

    def test_direction_tests_available_and_clustered(self):
        result = self.run_analysis()
        for trait, block in result["direction_tests"].items():
            with self.subTest(trait=trait):
                self.assertTrue(block["available"])
                self.assertGreaterEqual(block["n_persons"], 2)
                self.assertIn("p", block["common_direction_shift"])
                for key in ("group_heterogeneity_p", "wave_heterogeneity_p",
                            "cell_heterogeneity_p"):
                    self.assertIn(key, block)

    def test_limits_parameters(self):
        with self.assertRaises(ValueError):
            response_style_analysis(self.tables, {"bootstrap": 10})
        with self.assertRaises(ValueError):
            response_style_analysis(self.tables, {"unknown": 1})
        with self.assertRaises(ValueError):
            response_style_analysis(self.tables, {"group_by": ["person"]})
        with self.assertRaises(ValueError):
            response_style_analysis(self.tables, {"traits": ["missing"]})
        with self.assertRaises(ValueError):
            response_style_analysis(self.tables, {"wave_pairs": [[0, 99]]})

    def test_limited_release_is_insufficient_not_success(self):
        limited = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["limited"]
        validate(limited)
        result = response_style_analysis(limited, SMALL)
        for trait, block in result["reliability"].items():
            self.assertIn(block["status"], {"insufficient", "inadequate"})
            self.assertEqual(result["conclusions"][trait]["verdict"], "not_supported")

    def test_single_trait_and_wave_pair_subset(self):
        result = self.run_analysis(traits=["confidence"], wave_pairs=[[0, 2]])
        self.assertEqual(list(result["reliability"]), ["confidence"])
        self.assertEqual(result["design"]["wave_pairs"], [[0, 2]])
        for entry in result["changes"]["confidence"]:
            self.assertEqual(entry["waves"], [0, 2])

    def test_defaults_are_documented_and_consistent(self):
        self.assertIn("bootstrap", DEFAULTS)
        self.assertIn("holdout_fraction", DEFAULTS)
        self.assertIn("reliability_threshold", DEFAULTS)

    def test_identification_is_reported(self):
        result = self.run_analysis()
        self.assertTrue(result["design"]["identification"])
        self.assertTrue(result["limitations"])
        for trait, conclusion in result["conclusions"].items():
            self.assertIn("dependencies", conclusion)
            self.assertIn("items", conclusion["dependencies"])
            self.assertIn("anchors", conclusion["dependencies"])
            self.assertIn("constraints", conclusion["dependencies"])


if __name__ == "__main__":
    unittest.main()


class ResponseStyleSignalTests(unittest.TestCase):
    """构造有真实内容的合成数据，验证分析不是只会返回“不支持”。"""

    @staticmethod
    def synthetic(seed=7):
        import numpy as np
        rng = np.random.default_rng(seed)
        people = []
        for site in range(4):
            for group in ("new", "returning"):
                for _ in range(10):
                    people.append({"person": f"p{len(people):03d}", "site": f"s{site}",
                                   "group": group, "age_band": "60-69"})
        spec = [("i0", "confidence", False), ("i1", "confidence", True),
                ("i2", "confidence", False), ("i3", "confidence", True),
                ("i4", "belonging", False), ("i5", "belonging", True),
                ("i6", "belonging", False), ("i7", "belonging", True)]
        items = [{"item": n, "trait": t, "reverse": r, "minimum": 1, "maximum": 5}
                 for n, t, r in spec]
        person_effect = {p["person"]: rng.normal(0, 1.0) for p in people}
        answers = []
        for person in people:
            for wave in (0, 1, 2):
                for name, trait, reverse in spec:
                    center = 3.0 + 0.3 * wave + person_effect[person["person"]]
                    value = rng.normal(center, 0.6)
                    if reverse:
                        value = 6 - value
                    answers.append({"person": person["person"], "wave": wave,
                                    "item": name,
                                    "answer": int(np.clip(round(value), 1, 5))})
        vignettes = [{"vignette": "v0", "trait": "confidence", "order": 0},
                     {"vignette": "v1", "trait": "confidence", "order": 1},
                     {"vignette": "v2", "trait": "belonging", "order": 0},
                     {"vignette": "v3", "trait": "belonging", "order": 1}]
        vignette_answers = []
        for person in people:
            for wave in (0, 1, 2):
                for low, high in (("v0", "v1"), ("v2", "v3")):
                    center = 3.0 + 0.3 * wave + person_effect[person["person"]]
                    for vignette, offset in ((low, -0.4), (high, 0.4)):
                        value = int(np.clip(round(rng.normal(center + offset, 0.5)), 1, 5))
                        vignette_answers.append({"person": person["person"], "wave": wave,
                                                 "vignette": vignette, "answer": value})
        return {"people": people, "items": items, "answers": answers,
                "vignettes": vignettes, "vignette_answers": vignette_answers}

    def test_content_signal_is_detected(self):
        tables = self.synthetic()
        result = response_style_analysis(tables, {"bootstrap": 99, "permutations": 99,
                                                  "folds": 3})
        for trait in ("confidence", "belonging"):
            with self.subTest(trait=trait):
                self.assertEqual(result["reliability"][trait]["status"], "adequate")
                self.assertTrue(result["comparability"][trait]["comparable"])
                self.assertEqual(result["conclusions"][trait]["verdict"], "qualified")
                self.assertEqual(result["vignettes"][trait]["status"], "supported")

    def test_wave_change_is_supported_with_signal(self):
        tables = self.synthetic()
        result = response_style_analysis(tables, {"bootstrap": 99, "permutations": 99,
                                                  "folds": 3})
        supported = [verdict for verdict in result["change_verdicts"]["confidence"]
                     if verdict["measure"] == "content" and verdict["waves"] == [0, 2]
                     and verdict["level"].get("site") == "s0"]
        self.assertTrue(supported)
        self.assertTrue(all(verdict["supported"] for verdict in supported))
        self.assertTrue(all(verdict["ci"][0] > 0 for verdict in supported))


class ResponseStyleMechanismTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_min_items_per_direction_is_enforced(self):
        result = response_style_analysis(self.tables, {
            **SMALL, "min_items_per_direction": 5})
        for trait, block in result["reliability"].items():
            self.assertEqual(block["status"], "insufficient")
            self.assertFalse(result["comparability"][trait]["comparable"])

    def test_style_tolerance_is_reported(self):
        tight = response_style_analysis(self.tables, {**SMALL, "style_tolerance": 0.0})
        loose = response_style_analysis(self.tables, {**SMALL, "style_tolerance": 0.5})
        for trait in tight["comparability"]:
            with self.subTest(trait=trait):
                tight_row = [c for c in tight["comparability"][trait]["constraints"]
                             if c["constraint"] == "style_level_spread"][0]
                loose_row = [c for c in loose["comparability"][trait]["constraints"]
                             if c["constraint"] == "style_level_spread"][0]
                self.assertEqual(tight_row["status"], "warn")
                self.assertEqual(loose_row["status"], "pass")

    def test_content_changes_exclude_unbalanced_cells(self):
        result = response_style_analysis(self.tables, SMALL)
        for trait, block in result["changes"].items():
            for entry in block:
                if entry["measure"] == "content":
                    self.assertIn("n_excluded_unbalanced", entry["overall"])
                    self.assertGreaterEqual(entry["overall"]["n_excluded_unbalanced"], 0)

    def test_forward_and_reverse_only_sensitivity_are_estimable(self):
        result = response_style_analysis(self.tables, SMALL)
        for trait, block in result["sensitivity"].items():
            rows = {row["scenario"]: row for row in block["scenarios"]}
            self.assertGreater(rows["forward_only"]["n_pairs"], 0)
            self.assertGreater(rows["reverse_only"]["n_pairs"], 0)
            self.assertFalse(rows["forward_only"]["same_sign_as_base"])

    def test_vignette_convergence_requires_significance(self):
        result = response_style_analysis(self.tables, SMALL)
        for trait, block in result["vignettes"].items():
            self.assertIn("vignette_scale_p", block)
            if block["vignette_scale_p"] is not None and block["vignette_scale_p"] >= 0.05:
                self.assertFalse(block["convergence_supported"])
