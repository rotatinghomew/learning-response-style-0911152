"""作答方式（response style）与跨组/跨轮可比性分析。

一、可识别量
对每个 trait，把题目按内容方向重新编码（反向题翻转）、按题目位置中心化后，
某人在某一轮若同时回答了正反向题，记

    f = 正向题平均（内容方向）
    r = 反向题平均（内容方向）

若正反向题的内容位置可比，可写成 f = theta + a + e_f，r = theta - a + e_r，
其中 theta 是体验内容，a 是默认赞同（acquiescence）。于是

    C = (f + r) / 2   平衡内容
    S = (f - r) / 2   方向偏差 / 作答方式

C 消除“加性且正反向相同”的默认赞同，因此组别或轮次的内容差值不需要先把各校区
校准到同一分布。S 描述当场作答方式，只有在同一学员跨轮重复出现时才可当作稳定特质。

二、识别约束（都显式登记并逐项检验）
1. 正反向题平均内容位置可比。若反向题附带额外理解负担 delta，则
   C = theta + (b_F + b_R)/2、S = a + (b_F - b_R)/2。共同负担只平移 C 的绝对
   水平，组别/轮次差值仍可比；随组别或轮次变化的负担会破坏可比性。
2. 题目×组别、题目×轮次的分布不变量成立，否则该题的类别含义随组别或轮次漂移。
3. 内容分数具备可重复信度；否则组间/轮次差值与噪声无法区分。
4. 固定情境必须先通过顺序预期与情境-量表收敛检验，才可作效标；否则不参与估计，
   更不用于把校区强行校准到同一分布。
5. 观测为 1–5 有序等级；均值与差值按线性计分只作描述，不主张等距测量。

三、留出与可追溯
所有题目位置、信度、不变性、方向检验、变化估计、敏感性与选型只用 fit 学员。
整位 holdout 学员在选型完成后只做一次样本外预测检查，其记录不参与任何拟合或选型。
结果逐项报告结论依赖的题目、组别锚点、识别约束，以及去掉某题或某组、改用
正/反向题子集后结论是否仍成立。
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import stats

VERSION = "response_style/1"

GROUP_FIELDS = ("site", "group", "age_band")

DEFAULTS = {
    "group_by": ["site", "group"],
    "traits": None,
    "wave_pairs": None,
    "holdout_fraction": 0.25,
    "folds": 5,
    "seed": 20261115,
    "bootstrap": 299,
    "permutations": 299,
    "alpha": 0.05,
    "reliability_threshold": 0.5,
    "min_pairs": 8,
    "min_items_per_direction": 1,
    "max_missing_rate": 0.5,
    "style_tolerance": 0.1,
}

PARAMETER_KEYS = set(DEFAULTS)

CANDIDATE_ORDER = ("item", "direct", "content", "style", "vignette")
CANDIDATE_DESCRIPTION = {
    "item": "只用题目位置（内容方向总体均值）；组别与轮次无差异的最低基线",
    "direct": "题目位置 + 组×轮；直接比较观察分数，不区分正反向",
    "content": "题目位置 + 组×轮 + 方向；假设方向负担只造成共同平移",
    "style": "题目位置 + 组×轮 + 方向×组×轮；允许方向负担随组别与轮次变化",
    "vignette": "用通过检验的固定情境均值作为内容预测",
}

LIMITATIONS = [
    "作答是 1 至 5 的有序等级；均值与差值按线性计分，不主张等距测量。"
    "顺序与分布检验只使用符号、秩与类别频数。",
    "平衡内容 C 只能消除正反向共同的加性作答方式；随组别或轮次变化的反向题"
    "理解负担与真实内容变化在观测上部分等价，只能由逐题不变性检验和方向×组/轮"
    "检验间接核查。",
    "固定情境的 order 只在同一 trait 内表示预期顺序，不给出距离；情境不通过顺序"
    "或收敛检验时不会被当作锚，也不会用于校准。",
    "缺行按未答处理，不做插补，也不把缺失当作中性答案。",
    "区间来自对 fit 学员的整位重抽样；它们描述抽样不确定性，不消除上述识别限制。",
    "多个组别/轮次对比是探索性的；逐题不变性与方向检验使用 Holm 校正，其余为"
    "未校正的描述性区间。",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    values = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _sd(values):
    values = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _ci(values):
    values = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    if len(values) < 2:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [_finite(low), _finite(high)]


def _corr(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 4:
        return None, len(pairs)
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None, len(pairs)
    return _finite(np.corrcoef(x, y)[0, 1]), len(pairs)


def _spearman_brown(r):
    if r is None or abs(1.0 + r) <= 1e-12:
        return None
    return _finite(2.0 * r / (1.0 + r))


def _binomial_greater(successes, trials):
    if trials < 1:
        return None
    return _finite(stats.binomtest(int(successes), int(trials), 0.5,
                                   alternative="greater").pvalue)


def _holm(p_values):
    """Holm-Bonferroni 调整；None 原样保留，不参与排序。"""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adjusted = [None] * len(p_values)
    ordered = sorted(indexed, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, p) in enumerate(ordered):
        value = min(1.0, max(running, (total - rank) * p))
        adjusted[index] = _finite(value)
        running = value
    return adjusted


def _normalize(raw, item):
    return (float(raw) - item.minimum) / item.span


def _content_value(item, raw):
    value = _normalize(raw, item)
    return 1.0 - value if item.reverse else value


def _fold(person, seed, stage, folds):
    value = hashlib.sha256(f"{seed}|{stage}|{person}".encode("utf-8")).hexdigest()
    return int(value, 16) % folds


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_parameters(tables, parameters):
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError("分析参数不是对象")
    unknown = set(parameters) - PARAMETER_KEYS
    if unknown:
        raise ValueError(f"未知分析参数：{sorted(unknown)}")
    config = dict(DEFAULTS)
    config.update(parameters)

    group_by = config["group_by"]
    if (not isinstance(group_by, list) or not group_by
            or len(set(group_by)) != len(group_by)):
        raise ValueError("group_by 必须是非空且不重复的字段列表")
    people_fields = set(tables["people"][0].keys()) if tables["people"] else set()
    for name in group_by:
        if name not in GROUP_FIELDS or name not in people_fields:
            raise ValueError(f"group_by 字段不可用：{name}")

    traits_available = sorted({row["trait"] for row in tables["items"]})
    traits = config["traits"]
    if traits is None:
        traits = traits_available
    if not isinstance(traits, list) or not traits or len(set(traits)) != len(traits):
        raise ValueError("traits 必须是非空且不重复的列表")
    for name in traits:
        if name not in traits_available:
            raise ValueError(f"未知 trait：{name}")
    config["traits"] = list(traits)

    waves = sorted({row["wave"] for row in tables["answers"]} |
                   {row["wave"] for row in tables["vignette_answers"]})
    pairs = config["wave_pairs"]
    if pairs is None:
        pairs = [[waves[i], waves[j]] for i in range(len(waves))
                 for j in range(i + 1, len(waves))]
    if not isinstance(pairs, list):
        raise ValueError("wave_pairs 必须是列表")
    cleaned = []
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or pair[0] not in waves or pair[1] not in waves or pair[0] == pair[1]):
            raise ValueError("wave_pairs 必须引用存在的两个不同轮次")
        cleaned.append([pair[0], pair[1]] if pair[0] < pair[1] else [pair[1], pair[0]])
    config["wave_pairs"] = cleaned
    config["waves"] = waves

    def number(name, low, high, integer=False):
        value = config[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} 不是有限数值")
        if not low <= value <= high:
            raise ValueError(f"{name} 超出允许范围")
        return int(value) if integer else float(value)

    config["holdout_fraction"] = number("holdout_fraction", 0.05, 0.6)
    config["folds"] = number("folds", 2, 10, True)
    config["bootstrap"] = number("bootstrap", 50, 5000, True)
    config["permutations"] = number("permutations", 50, 5000, True)
    config["alpha"] = number("alpha", 0.001, 0.2)
    config["reliability_threshold"] = number("reliability_threshold", 0.0, 0.95)
    config["min_pairs"] = number("min_pairs", 3, 1000, True)
    config["min_items_per_direction"] = number("min_items_per_direction", 1, 20, True)
    config["max_missing_rate"] = number("max_missing_rate", 0.0, 0.95)
    config["style_tolerance"] = number("style_tolerance", 0.0, 0.5)
    config["seed"] = number("seed", 0, 2 ** 31 - 1, True)
    return config


# ---------------------------------------------------------------------------
# 数据整理
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    name: str
    trait: str
    reverse: bool
    minimum: float
    maximum: float

    @property
    def span(self):
        return self.maximum - self.minimum


@dataclass
class Evidence:
    people: dict
    items: dict
    answers: dict
    vignettes: dict
    vignette_answers: dict
    waves: list
    persons: list
    fit_persons: list
    holdout_persons: list
    group_levels: dict


def build_evidence(tables, config):
    people = {row["person"]: dict(row) for row in tables["people"]}
    items = {row["item"]: Item(row["item"], row["trait"], bool(row["reverse"]),
                               float(row["minimum"]), float(row["maximum"]))
             for row in tables["items"]}
    answers = defaultdict(dict)
    for row in tables["answers"]:
        answers[(row["person"], row["wave"])][row["item"]] = row["answer"]
    vignettes = {row["vignette"]: {"trait": row["trait"], "order": row["order"]}
                 for row in tables["vignettes"]}
    vignette_answers = defaultdict(dict)
    for row in tables["vignette_answers"]:
        vignette_answers[(row["person"], row["wave"])][row["vignette"]] = row["answer"]
    waves = sorted({row["wave"] for row in tables["answers"]} |
                   {row["wave"] for row in tables["vignette_answers"]})
    persons = sorted(people)

    hashed = {p: _fold(p, config["seed"], "holdout", 1000) for p in persons}
    threshold = int(round(config["holdout_fraction"] * 1000))
    holdout = [p for p in persons if hashed[p] < threshold]
    if not holdout or len(holdout) >= len(persons):
        ordered = sorted(persons, key=lambda p: hashlib.sha256(
            f"{config['seed']}|order|{p}".encode("utf-8")).hexdigest())
        size = min(max(int(round(config["holdout_fraction"] * len(persons))), 2),
                   max(len(persons) - 2, 1))
        holdout = sorted(ordered[:size])
    holdout_set = set(holdout)
    fit = [p for p in persons if p not in holdout_set]
    if len(fit) < 4:
        raise ValueError("可用学员不足以同时留出与拟合")

    group_levels = {}
    for field in config["group_by"]:
        levels = sorted({people[p][field] for p in persons})
        if len(levels) < 2:
            raise ValueError(f"分组字段 {field} 少于两个水平")
        group_levels[field] = levels

    return Evidence(people=people, items=items, answers=dict(answers),
                    vignettes=vignettes, vignette_answers=dict(vignette_answers),
                    waves=waves, persons=persons, fit_persons=fit,
                    holdout_persons=sorted(holdout), group_levels=group_levels)


def item_locations(evidence, persons):
    """fit 学员上、内容方向的题目位置（0–1）。"""
    totals = defaultdict(list)
    for person in persons:
        for wave in evidence.waves:
            for name, raw in sorted(evidence.answers.get((person, wave), {}).items()):
                totals[name].append(_content_value(evidence.items[name], raw))
    return {name: float(np.mean(values)) for name, values in totals.items() if values}


def cell_indices(evidence, persons, locations, traits):
    """返回 {(person, wave, trait): 单元格指标}。

    C 为平衡内容；S 为方向偏差，仅在正反向题都作答时可算。
    """
    ordered = sorted(evidence.items)
    forward = {trait: [n for n in ordered
                       if evidence.items[n].trait == trait and not evidence.items[n].reverse]
               for trait in traits}
    reverse = {trait: [n for n in ordered
                       if evidence.items[n].trait == trait and evidence.items[n].reverse]
               for trait in traits}
    cells = {}
    for person in persons:
        for wave in evidence.waves:
            answered = {name: raw for name, raw in sorted(
                evidence.answers.get((person, wave), {}).items())
                if name in evidence.items and name in locations}
            if not answered:
                continue
            centered = {name: _content_value(evidence.items[name], raw) - locations[name]
                        for name, raw in sorted(answered.items())}
            for trait in traits:
                f = [centered[n] for n in forward[trait] if n in centered]
                r = [centered[n] for n in reverse[trait] if n in centered]
                if not f and not r:
                    continue
                f_mean = float(np.mean(f)) if f else None
                r_mean = float(np.mean(r)) if r else None
                if f_mean is not None and r_mean is not None:
                    content = (f_mean + r_mean) / 2.0
                    style = (f_mean - r_mean) / 2.0
                else:
                    content = f_mean if f_mean is not None else r_mean
                    style = None
                extremes = [1.0 if answered[n] in (evidence.items[n].minimum,
                                                   evidence.items[n].maximum) else 0.0
                            for n in answered]
                midpoints = [1.0 if abs(_normalize(answered[n], evidence.items[n]) - 0.5) <= 0.125
                             else 0.0 for n in answered]
                forward_extremes = [1.0 if answered[n] in (evidence.items[n].minimum,
                                                          evidence.items[n].maximum) else 0.0
                                    for n in forward[trait] if n in answered]
                reverse_extremes = [1.0 if answered[n] in (evidence.items[n].minimum,
                                                          evidence.items[n].maximum) else 0.0
                                    for n in reverse[trait] if n in answered]
                midpoint_raw = {n: evidence.items[n].minimum + evidence.items[n].span / 2.0
                                for n in answered}
                cells[(person, wave, trait)] = {
                    "f": f_mean, "r": r_mean, "C": content, "S": style,
                    "n_forward": len(f), "n_reverse": len(r),
                    "n_items": len(f) + len(r),
                    "balanced": style is not None,
                    "raw_mean": float(np.mean(list(answered.values()))),
                    "extreme": float(np.mean(extremes)) if extremes else None,
                    "midpoint": float(np.mean(midpoints)) if midpoints else None,
                    "forward_extreme": float(np.mean(forward_extremes))
                    if forward_extremes else None,
                    "reverse_extreme": float(np.mean(reverse_extremes))
                    if reverse_extremes else None,
                    "forward_midpoint": (float(np.mean(
                        [1.0 if answered[n] == midpoint_raw[n] else 0.0
                         for n in forward[trait] if n in answered]))
                        if any(n in answered for n in forward[trait]) else None),
                    "reverse_midpoint": (float(np.mean(
                        [1.0 if answered[n] == midpoint_raw[n] else 0.0
                         for n in reverse[trait] if n in answered]))
                        if any(n in answered for n in reverse[trait]) else None),
                    "answered_items": sorted(answered),
                }
    return cells


def _cell_key(evidence, person, wave, group_by):
    return tuple(evidence.people[person][field] for field in group_by) + (wave,)


# ---------------------------------------------------------------------------
# 信度与作答方式结构
# ---------------------------------------------------------------------------

def _icc(person_values):
    groups = [v for v in person_values.values() if v]
    count = len(groups)
    total = sum(len(v) for v in groups)
    if count < 3 or total <= count:
        return None
    grand = float(np.mean([y for v in groups for y in v]))
    between = sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in groups) / (count - 1)
    within = sum(sum((float(y) - float(np.mean(v))) ** 2 for y in v) for v in groups) / (total - count)
    n0 = (total - sum(len(v) ** 2 for v in groups) / total) / (count - 1)
    if n0 <= 0:
        return None
    between = max(0.0, (between - within) / n0)
    return _finite(between / (between + within)) if between + within > 0 else None


def _bootstrap_values(rng, values, count, statistic):
    if len(values) < 4:
        return []
    outputs = []
    for _ in range(count):
        index = rng.integers(0, len(values), len(values))
        try:
            value = statistic([values[i] for i in index])
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if value is not None and math.isfinite(value):
            outputs.append(float(value))
    return outputs


def reliability(evidence, persons, trait, config):
    rng = np.random.default_rng([config["seed"], 11])
    items = sorted(n for n, item in evidence.items.items() if item.trait == trait)
    forward = [n for n in items if not evidence.items[n].reverse]
    reverse = [n for n in items if evidence.items[n].reverse]

    complete = []
    for person in persons:
        for wave in evidence.waves:
            answered = evidence.answers.get((person, wave), {})
            if all(n in answered for n in items):
                complete.append([_content_value(evidence.items[n], answered[n]) for n in items])

    alpha = split_r = split_sb = style_sb = None
    n_complete = len(complete)
    if n_complete >= 4 and len(items) >= 2:
        matrix = np.array(complete, dtype=float)
        k = matrix.shape[1]
        total_var = float(np.var(matrix.sum(axis=1), ddof=1))
        if total_var > 1e-12:
            alpha = _finite(k / (k - 1) * (1 - float(np.var(matrix, axis=0, ddof=1).sum())
                                           / total_var))
        if forward and reverse:
            f_scores = matrix[:, [items.index(n) for n in forward]].mean(axis=1)
            r_scores = matrix[:, [items.index(n) for n in reverse]].mean(axis=1)
            split_r, _ = _corr(f_scores.tolist(), r_scores.tolist())
            split_sb = _spearman_brown(split_r)

    if len(forward) >= 2 and len(reverse) >= 2:
        first, second = [], []
        for person in persons:
            for wave in evidence.waves:
                answered = evidence.answers.get((person, wave), {})
                if all(n in answered for n in items):
                    values = {n: _content_value(evidence.items[n], answered[n]) for n in items}
                    first.append((values[forward[0]] - values[reverse[0]]) / 2.0)
                    second.append((values[forward[1]] - values[reverse[1]]) / 2.0)
        r, _ = _corr(first, second)
        style_sb = _spearman_brown(r)

    def alpha_statistic(sample):
        matrix = np.array(sample, dtype=float)
        k = matrix.shape[1]
        total = float(np.var(matrix.sum(axis=1), ddof=1))
        if total <= 1e-12:
            return None
        return k / (k - 1) * (1 - float(np.var(matrix, axis=0, ddof=1).sum()) / total)

    alpha_draws = _bootstrap_values(rng, complete, config["bootstrap"], alpha_statistic)
    minimum = max(4, config["min_pairs"])
    reasons = []
    if len(forward) < config["min_items_per_direction"]:
        status = "insufficient"
        reasons.append(f"正向题只有 {len(forward)} 道，少于 "
                       f"{config['min_items_per_direction']}")
    elif len(reverse) < config["min_items_per_direction"]:
        status = "insufficient"
        reasons.append(f"反向题只有 {len(reverse)} 道，少于 "
                       f"{config['min_items_per_direction']}")
    elif n_complete < minimum:
        status = "insufficient"
        reasons.append(f"完整作答单元只有 {n_complete} 个，少于 {minimum}")
    elif split_sb is None:
        status = "inadequate"
        reasons.append("无法计算正反向分半信度")
    elif split_sb < config["reliability_threshold"]:
        status = "inadequate"
        reasons.append(
            f"正反向分半信度 {split_sb:.3f} 低于阈值 {config['reliability_threshold']:.2f}；"
            "内容分数中可重复的信号不足以把组间或轮次差值与测量噪声区分开")
    else:
        status = "adequate"
    if alpha is not None and alpha <= 0:
        reasons.append(f"Cronbach alpha={alpha:.3f} 不高于 0，题目之间没有可重复的共同变异")
    if split_sb is not None and split_sb < 0:
        reasons.append(f"正反向题重编码后相关为 {split_r:.3f}，方向相反；"
                       "这比单纯噪声更差，提示方向编码或题目理解存在系统问题")
    complete_persons = len({
        person for person in persons
        if any(all(n in evidence.answers.get((person, wave), {}) for n in items)
               for wave in evidence.waves)})
    return {
        "trait": trait,
        "status": status,
        "n_complete_cells": n_complete,
        "n_complete_persons": complete_persons,
        "cronbach_alpha": alpha,
        "cronbach_alpha_ci": _ci(alpha_draws),
        "split_half_forward_reverse_r": split_r,
        "split_half_forward_reverse_spearman_brown": split_sb,
        "style_split_half_spearman_brown": style_sb,
        "items": {"forward": forward, "reverse": reverse},
        "reasons": reasons,
    }


def style_structure(evidence, cells, persons, trait, config):
    rng = np.random.default_rng([config["seed"], 12])
    style_by_person = defaultdict(list)
    content_by_person = defaultdict(list)
    style_by_wave = defaultdict(dict)
    for person in persons:
        for wave in evidence.waves:
            cell = cells.get((person, wave, trait))
            if not cell:
                continue
            if cell["S"] is not None:
                style_by_person[person].append(cell["S"])
                style_by_wave[wave][person] = cell["S"]
            if cell["C"] is not None:
                content_by_person[person].append(cell["C"])

    style_icc = _icc(style_by_person)
    content_icc = _icc(content_by_person)
    style_draws = _bootstrap_values(
        rng, [style_by_person[p] for p in sorted(style_by_person)],
        config["bootstrap"], lambda sample: _icc({i: v for i, v in enumerate(sample)}))

    correlations = {}
    for i, first in enumerate(evidence.waves):
        for second in evidence.waves[i + 1:]:
            common = [p for p in persons if p in style_by_wave[first]
                      and p in style_by_wave[second]]
            xs = [style_by_wave[first][p] for p in common]
            ys = [style_by_wave[second][p] for p in common]
            r, n_pairs = _corr(xs, ys)
            draws = []
            if n_pairs >= 4:
                pairs = list(zip(xs, ys))
                draws = _bootstrap_values(
                    rng, pairs, config["bootstrap"],
                    lambda sample: _corr([s[0] for s in sample],
                                         [s[1] for s in sample])[0])
            correlations[f"{first}->{second}"] = {"r": r, "n": n_pairs, "ci": _ci(draws)}

    stable = (style_icc is not None and style_icc >= config["reliability_threshold"]
              and all(v["r"] is not None and v["r"] > 0 for v in correlations.values()))
    return {
        "trait": trait,
        "style_icc": style_icc,
        "style_icc_ci": _ci(style_draws),
        "content_icc": content_icc,
        "cross_wave_style": correlations,
        "stable_style_trait": bool(stable),
        "note": ("作答方式只有在同一学员跨轮重复出现（ICC 与跨轮相关明显为正）时，"
                 "才可作为稳定特质；否则 S 只描述当场作答，不能作为把不同组别校准"
                 "到同一分布的个体校准量。"),
    }


def style_levels(evidence, cells, persons, trait, config):
    """各组×轮次的方向偏差与作答方式水平，含整位 bootstrap 区间。"""
    rng = np.random.default_rng([config["seed"], 13])
    group_by = config["group_by"]
    cells_seen = sorted({_cell_key(evidence, p, w, group_by)
                         for p in persons for w in evidence.waves
                         if (p, w, trait) in cells})
    output = []
    for key in cells_seen:
        style = [cells[(p, w, trait)]["S"] for p in persons for w in evidence.waves
                 if _cell_key(evidence, p, w, group_by) == key
                 and (p, w, trait) in cells and cells[(p, w, trait)]["S"] is not None]
        raw = [cells[(p, w, trait)]["raw_mean"] for p in persons for w in evidence.waves
               if _cell_key(evidence, p, w, group_by) == key
               and (p, w, trait) in cells]
        extremes = [cells[(p, w, trait)]["extreme"] for p in persons for w in evidence.waves
                    if _cell_key(evidence, p, w, group_by) == key
                    and (p, w, trait) in cells and cells[(p, w, trait)]["extreme"] is not None]
        by_person = {}
        for p in persons:
            values = [cells[(p, w, trait)]["S"] for w in evidence.waves
                      if _cell_key(evidence, p, w, group_by) == key
                      and (p, w, trait) in cells and cells[(p, w, trait)]["S"] is not None]
            if values:
                by_person[p] = float(np.mean(values))
        draws = _bootstrap_values(rng, list(by_person.values()), config["bootstrap"], _mean)
        output.append({
            "level": dict(zip(group_by, key[:-1])) | {"wave": key[-1]},
            "n_observations": len(style),
            "n_persons": len(by_person),
            "mean_style": _mean(style),
            "mean_style_ci": _ci(draws),
            "mean_raw": _mean(raw),
            "mean_extreme": _mean(extremes),
            "note": "style 为 f-r 的一半，正值表示正向题相对反向题更偏向高分。",
        })
    return output


# ---------------------------------------------------------------------------
# 逐题测量等价（DIF）检验：题目 × 组别、题目 × 轮次
# ---------------------------------------------------------------------------
#
# 直接把某题的边际作答分布按组别或轮次比较会把“真实内容差异”误判为测量不等价。
# 这里改为条件在单元格平衡内容 C 上做逐题 DIF：
#
#   residual_ij = 内容方向分数_ij - C_cell
#
# 在平衡量尺下，residual 的组别/轮次均值应为 0，与真实内容水平无关。若某题在某个
# 组别或轮次的 residual 均值显著偏离 0，说明该题的位置随组别/轮次漂移（非均匀
# DIF）。若反向题的 residual 在组别/轮次间呈相反方向的漂移，则由方向检验单独
# 报告为负担异质性。
#
# 回归按学员聚类，保留同一学员跨轮重复记录；每题对组别、轮次的 Wald 检验分别做
# Holm 校正。

def _dif_values(evidence, cells, persons, trait):
    """{(person, wave, item): 内容方向分数 − 该单元格平衡内容 C}。"""
    values = {}
    for person in persons:
        for wave in evidence.waves:
            cell = cells.get((person, wave, trait))
            if not cell or cell["C"] is None:
                continue
            answered = evidence.answers.get((person, wave), {})
            for name, raw in answered.items():
                if evidence.items[name].trait != trait:
                    continue
                values[(person, wave, name)] = (
                    _content_value(evidence.items[name], raw) - cell["C"])
    return values


def _dif_person_means(values, item):
    """{person: 该题残差按人平均}，每次置换只重排学员标签，不必重算。"""
    samples = defaultdict(list)
    for (person, wave, name), value in values.items():
        if name == item:
            samples[person].append(value)
    return {person: float(np.mean(v)) for person, v in samples.items()}


def _dif_group_statistic(person_means, labels):
    groups = defaultdict(list)
    for person, value in person_means.items():
        groups[labels[person]].append(value)
    return _weighted_between(groups)


def _dif_wave_statistic(values, item):
    groups = defaultdict(list)
    for (person, wave, name), value in values.items():
        if name == item:
            groups[wave].append(value)
    return _weighted_between(groups)


def dif_tests(evidence, cells, persons, traits, config):
    """逐题测量等价（DIF）检验，按学员聚类并采用置换推断。

    条件在单元格平衡内容 C 上：residual = 内容方向分数 − C。若题目在平衡量尺下
    位置不变，residual 的组别/轮次均值应为 0，且不随真实内容水平变化。
    - 组别 DIF：在学员之间置换登记分组标签，比较按人平均残差的组间差异；
    - 轮次 DIF：在同一学员内部置换轮次标签。
    置换在零假设下精确，避免少聚类下的 Wald 过度拒绝；逐题 p 值经 Holm 校正。
    """
    rng = np.random.default_rng([config["seed"], 23])
    group_by = config["group_by"]
    output = {}
    for trait in traits:
        items = sorted(name for name, item in evidence.items.items() if item.trait == trait)
        values = _dif_values(evidence, cells, persons, trait)
        usable = sorted({key[0] for key in values})
        labels = {p: tuple(evidence.people[p][field] for field in group_by) for p in persons}
        entries = []
        for name in items:
            relevant = [p for p in usable
                        if any(key[0] == p and key[2] == name for key in values)]
            if len(relevant) < 8:
                entries.append({"item": name, "reverse": evidence.items[name].reverse,
                                "available": False, "reason": "有效观测不足"})
                continue
            person_means = _dif_person_means(values, name)
            relevant = sorted(person_means)
            observed_group = _dif_group_statistic(person_means, labels)
            exceed_group = 0
            for _ in range(config["permutations"]):
                order = rng.permutation(len(relevant))
                shuffled = {p: labels[relevant[i]] for p, i in zip(relevant, order)}
                if _dif_group_statistic(person_means, shuffled) >= observed_group - 1e-12:
                    exceed_group += 1
            group_p = (exceed_group + 1) / (config["permutations"] + 1)

            observed_wave = _dif_wave_statistic(values, name)
            sources_by_person = {person: [wave for wave in evidence.waves
                                          if (person, wave, name) in values]
                                 for person in relevant}
            exceed_wave = 0
            for _ in range(config["permutations"]):
                statistics = defaultdict(list)
                for person in relevant:
                    sources = sources_by_person[person]
                    assigned = rng.permutation(sources)
                    for target, source in zip(sources, assigned):
                        statistics[target].append(values[(person, source, name)])
                if _weighted_between(statistics) >= observed_wave - 1e-12:
                    exceed_wave += 1
            wave_p = (exceed_wave + 1) / (config["permutations"] + 1)
            entries.append({
                "item": name,
                "reverse": evidence.items[name].reverse,
                "available": True,
                "n_observations": sum(1 for key in values if key[2] == name),
                "n_persons": len(relevant),
                "group_dif_p": _finite(group_p),
                "wave_dif_p": _finite(wave_p),
                "mean_residual": _mean([value for key, value in values.items()
                                        if key[2] == name]),
            })
        available = [e for e in entries if e.get("available")]
        group_adjusted = _holm([e["group_dif_p"] for e in available])
        wave_adjusted = _holm([e["wave_dif_p"] for e in available])
        for entry, group_p, wave_p in zip(available, group_adjusted, wave_adjusted):
            entry["group_dif_p_holm"] = group_p
            entry["wave_dif_p_holm"] = wave_p
            entry["flag_group"] = group_p is not None and group_p < config["alpha"]
            entry["flag_wave"] = wave_p is not None and wave_p < config["alpha"]

        group_levels = sorted(set(labels.values()))
        missing_items = []
        for name in items:
            rates = {}
            for group in group_levels:
                members = [p for p in persons if labels[p] == group]
                rates[str(group)] = _mean([
                    1.0 if name not in evidence.answers.get((p, wave), {}) else 0.0
                    for p in members for wave in evidence.waves]) if members else None
            values_here = [v for v in rates.values() if v is not None]
            if values_here and max(values_here) - min(values_here) >= config["max_missing_rate"]:
                missing_items.append(name)
        output[trait] = {
            "items": entries,
            "failed_group_items": [e["item"] for e in available if e["flag_group"]],
            "failed_wave_items": [e["item"] for e in available if e["flag_wave"]],
            "differential_missing_items": missing_items,
            "alpha": config["alpha"],
            "method": ("逐题残差 = 内容方向分数 − 该单元格平衡内容 C；在单元格内容水平"
                       "给定的条件下，用置换检验比较残差按人平均的组别差异及学员内轮次"
                       "差异。逐题 p 值经 Holm 校正；该检验不会把真实内容差异误判为 DIF。"),
        }
    return output


# ---------------------------------------------------------------------------
# 固定情境：先检验，再决定是否可用
# ---------------------------------------------------------------------------

def vignette_diagnostics(evidence, persons, traits, config):
    output = {}
    for trait in traits:
        block = [name for name, spec in evidence.vignettes.items() if spec["trait"] == trait]
        block.sort(key=lambda name: (evidence.vignettes[name]["order"], name))
        if len(block) < 2:
            output[trait] = {
                "trait": trait, "vignettes": block, "status": "unsupported",
                "reasons": ["同一 trait 下少于两个固定情境，无法检验顺序预期"],
                "pairs": [], "order_supported": False, "convergence_supported": False,
                "vignette_scale_correlation": None,
            }
            continue
        pairs = []
        for i, first in enumerate(block):
            for second in block[i + 1:]:
                differences = []
                for person in persons:
                    for wave in evidence.waves:
                        answers = evidence.vignette_answers.get((person, wave), {})
                        if first in answers and second in answers:
                            differences.append(answers[second] - answers[first])
                total = len(differences)
                positive = sum(1 for d in differences if d > 0)
                negative = sum(1 for d in differences if d < 0)
                p_greater = (_binomial_greater(positive, positive + negative)
                             if positive + negative else None)
                pairs.append({
                    "lower": first, "higher": second,
                    "n": total,
                    "mean_difference": _mean(differences),
                    "proportion_ordered": _finite(positive / total) if total else None,
                    "proportion_tied": (_finite((total - positive - negative) / total)
                                        if total else None),
                    "p_greater": p_greater,
                    "supported": bool(p_greater is not None and p_greater < config["alpha"]
                                      and (_mean(differences) or 0) > 0),
                })

        common = [p for p in persons for w in evidence.waves
                  if all(v in evidence.vignette_answers.get((p, w), {}) for v in block)]
        pair_cor, pair_n = _corr(
            [evidence.vignette_answers[(p, w)][block[0]] for p in persons
             for w in evidence.waves
             if all(v in evidence.vignette_answers.get((p, w), {}) for v in block)],
            [evidence.vignette_answers[(p, w)][block[1]] for p in persons
             for w in evidence.waves
             if all(v in evidence.vignette_answers.get((p, w), {}) for v in block)])
        scale_pairs = [(np.mean([evidence.vignette_answers[(p, w)][v] for v in block]),
                        _scale_content(evidence, p, w, trait))
                       for p in persons for w in evidence.waves
                       if all(v in evidence.vignette_answers.get((p, w), {}) for v in block)]
        scale_cor, scale_n = _corr([pair[0] for pair in scale_pairs],
                                   [pair[1] for pair in scale_pairs])
        scale_p = None
        if scale_cor is not None and scale_n > 3:
            # 单侧检验：情境均值与量表内容应正向收敛
            statistic = scale_cor * math.sqrt(max(scale_n - 2, 1)) / math.sqrt(
                max(1 - scale_cor ** 2, 1e-12))
            scale_p = _finite(stats.t.sf(statistic, scale_n - 2))

        order_supported = all(pair["supported"] for pair in pairs)
        convergence_supported = bool(scale_cor is not None and scale_cor > 0
                                     and scale_p is not None
                                     and scale_p < config["alpha"])
        reasons = []
        if not order_supported:
            failed = [pair for pair in pairs if not pair["supported"]]
            reasons.append("情境顺序未得到支持：" + "；".join(
                f"{pair['lower']}→{pair['higher']} 的提升比例 "
                f"{pair['proportion_ordered'] if pair['proportion_ordered'] is not None else '不可估计'}"
                for pair in failed))
        if scale_cor is None:
            reasons.append("情境均值与量表内容无法计算相关（有效配对不足）")
        elif scale_cor <= 0:
            reasons.append(f"情境均值与量表内容相关为 {scale_cor:.3f}，未表现出正向收敛")
        elif not convergence_supported:
            reasons.append(f"情境均值与量表内容相关为 {scale_cor:.3f}（p={scale_p:.3f}），"
                           "与噪声无法区分")
        output[trait] = {
            "trait": trait,
            "vignettes": block,
            "status": "supported" if order_supported and convergence_supported else "unsupported",
            "order_supported": order_supported,
            "convergence_supported": convergence_supported,
            "pairs": pairs,
            "vignette_pair_correlation": pair_cor,
            "vignette_pair_n": pair_n,
            "vignette_scale_correlation": scale_cor,
            "vignette_scale_p": scale_p,
            "vignette_scale_n": scale_n,
            "reasons": reasons,
            "note": ("情境是效标，不是天然正确的锚。只有顺序预期与情境-量表收敛同时成立时，"
                     "才允许把情境均值作为内容预测进入选型；否则不参与估计，也不用于校准。"),
        }
    return output


def _scale_content(evidence, person, wave, trait):
    answered = evidence.answers.get((person, wave), {})
    values = [_content_value(evidence.items[name], raw)
              for name, raw in sorted(answered.items())
              if evidence.items[name].trait == trait]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# 变化量与不确定性
# ---------------------------------------------------------------------------
#
# 变化量在 fit 学员上按人配对计算：同一学员在两轮的 C（或 S）相减。区间来自
# 整位 bootstrap，因此保留同一学员跨轮的相关。组别层面的变化是同一组内学员的
# 配对差值平均，不是把两轮的横截面均值相减。

def _paired(evidence, cells, persons, trait, pair, field, group_by=None, level=None,
            require_balanced=True):
    """同一学员两轮的配对差值。

    内容 C 默认只在正反向都作答的平衡单元格上比较：一侧只有单向题时 C 含作答
    方式，与另一侧的平衡 C 不可比，因此排除并另行计数。方向 S 本身要求正反向
    齐备。敏感性分析里的“只用正向/反向题”场景会显式关闭该过滤，并登记其假设。
    """
    first, second = pair
    values = []
    for person in persons:
        if level is not None and tuple(evidence.people[person][f]
                                       for f in group_by) != level:
            continue
        a = cells.get((person, first, trait))
        b = cells.get((person, second, trait))
        if not a or not b or a[field] is None or b[field] is None:
            continue
        if field == "C" and require_balanced and not (a["balanced"] and b["balanced"]):
            continue
        values.append(b[field] - a[field])
    return values


def _mean_interval(rng, values, count):
    if len(values) < 3:
        return {"n_pairs": len(values), "change": _mean(values), "sd": _sd(values),
                "ci": None}
    draws = _bootstrap_values(rng, values, count, _mean)
    return {"n_pairs": len(values), "change": _mean(values), "sd": _sd(values),
            "ci": _ci(draws)}


def _unbalanced_pairs(evidence, cells, persons, trait, pair, group_by=None, level=None):
    first, second = pair
    count = 0
    for person in persons:
        if level is not None and tuple(evidence.people[person][f]
                                       for f in group_by) != level:
            continue
        a = cells.get((person, first, trait))
        b = cells.get((person, second, trait))
        if a and b and a["C"] is not None and b["C"] is not None \
                and not (a["balanced"] and b["balanced"]):
            count += 1
    return count


def changes(evidence, cells, persons, traits, config):
    rng = np.random.default_rng([config["seed"], 51])
    group_by = config["group_by"]
    output = {}
    for trait in traits:
        block = []
        levels = sorted({tuple(evidence.people[p][field] for field in group_by)
                         for p in persons})
        for pair in config["wave_pairs"]:
            for field, measure in (("C", "content"), ("S", "style")):
                rows = []
                for level in levels:
                    row = _mean_interval(
                        rng, _paired(evidence, cells, persons, trait, pair,
                                     field, group_by, level), config["bootstrap"])
                    if field == "C":
                        row["n_excluded_unbalanced"] = _unbalanced_pairs(
                            evidence, cells, persons, trait, pair, group_by, level)
                    rows.append(dict(row, level=dict(zip(group_by, level))))
                overall = _mean_interval(
                    rng, _paired(evidence, cells, persons, trait, pair, field),
                    config["bootstrap"])
                if field == "C":
                    overall["n_excluded_unbalanced"] = _unbalanced_pairs(
                        evidence, cells, persons, trait, pair)
                block.append({
                    "waves": pair, "measure": measure,
                    "overall": overall,
                    "groups": rows,
                })
        output[trait] = block
    return output


# ---------------------------------------------------------------------------
# 方向（反向题负担）检验
# ---------------------------------------------------------------------------
#
# 对每个学员的每个单元格，D = f - r = 2S。若反向题比正向题多负担 delta，
# 则 D 的总体水平会偏离 0（共同负担），而 D 随组别或轮次的变化意味着负担与
# 分组/时间缠结，平衡内容 C 的差值不能再解释为体验变化。
# 回归按学员聚类，保留同一学员的重复记录。

def _direction_values(evidence, cells, persons, trait):
    values = {}
    for person in persons:
        for wave in evidence.waves:
            cell = cells.get((person, wave, trait))
            if cell and cell["f"] is not None and cell["r"] is not None:
                values[(person, wave)] = cell["f"] - cell["r"]
    return values


def _weighted_between(groups):
    all_values = [x for v in groups.values() for x in v]
    if not all_values:
        return 0.0
    grand = float(np.mean(all_values))
    return sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in groups.values() if v)


def _group_direction_statistic(values, persons, labels):
    groups = defaultdict(list)
    for person in persons:
        samples = [value for (owner, wave), value in values.items() if owner == person]
        if samples:
            groups[labels[person]].append(float(np.mean(samples)))
    return _weighted_between(groups)


def _wave_direction_statistic(values):
    groups = defaultdict(list)
    for (person, wave), value in values.items():
        groups[wave].append(value)
    return _weighted_between(groups)


def _cell_direction_statistic(values, persons, labels, waves):
    groups = defaultdict(list)
    for (person, wave), value in values.items():
        groups[(labels[person], wave)].append(value)
    return _weighted_between(groups)


def direction_tests(evidence, cells, persons, traits, config):
    """方向负担（D = f − r = 2S）的总体水平与随组别/轮次变化的检验。

    使用置换检验，不做大样本 Wald 近似：
    - 组别异质性：在学员之间置换登记分组标签（保留同一学员的轮次结构）；
    - 轮次异质性：在同一学员内部置换轮次标签；
    - 组×轮异质性：在学员之间置换分组标签后比较 group×wave 单元格。
    共同平移另用学员聚类 bootstrap 估计区间。
    """
    rng = np.random.default_rng([config["seed"], 22])
    group_by = config["group_by"]
    output = {}
    for trait in traits:
        values = _direction_values(evidence, cells, persons, trait)
        usable = [p for p in persons if any(key[0] == p for key in values)]
        if len(usable) < 8 or len(values) < 12:
            output[trait] = {"trait": trait, "available": False,
                             "reason": "同时作答正反向题的单元格不足"}
            continue
        labels = {p: tuple(evidence.people[p][field] for field in group_by) for p in persons}
        label_list = [labels[p] for p in persons]
        by_person = defaultdict(list)
        for (person, wave), value in values.items():
            by_person[person].append(value)
        person_means = [float(np.mean(v)) for v in by_person.values() if v]
        bootstrap = _bootstrap_values(rng, person_means, config["bootstrap"], _mean)
        estimate = _mean(person_means)
        se = _sd(person_means) / math.sqrt(len(person_means)) if len(person_means) > 1 else None
        common_ci = _ci(bootstrap)
        common_p = None
        if se and se > 0:
            common_p = _finite(2 * stats.t.sf(abs(estimate / se), len(person_means) - 1))

        observed_group = _group_direction_statistic(values, usable, labels)
        exceed_group = 0
        for _ in range(config["permutations"]):
            order = rng.permutation(len(usable))
            shuffled = {p: labels[usable[i]] for p, i in zip(usable, order)}
            if _group_direction_statistic(values, usable, shuffled) >= observed_group - 1e-12:
                exceed_group += 1
        group_p = (exceed_group + 1) / (config["permutations"] + 1)

        observed_wave = _wave_direction_statistic(values)
        exceed_wave = 0
        for _ in range(config["permutations"]):
            permuted = {}
            for person in usable:
                sources = [w for w in evidence.waves if (person, w) in values]
                assigned = rng.permutation(sources)
                for target, source in zip(sources, assigned):
                    permuted[(person, target)] = values[(person, source)]
            if _wave_direction_statistic(permuted) >= observed_wave - 1e-12:
                exceed_wave += 1
        wave_p = (exceed_wave + 1) / (config["permutations"] + 1)

        observed_cell = _cell_direction_statistic(values, usable, labels, evidence.waves)
        exceed_cell = 0
        for _ in range(config["permutations"]):
            order = rng.permutation(len(usable))
            shuffled = {p: labels[usable[i]] for p, i in zip(usable, order)}
            if _cell_direction_statistic(values, usable, shuffled,
                                         evidence.waves) >= observed_cell - 1e-12:
                exceed_cell += 1
        cell_p = (exceed_cell + 1) / (config["permutations"] + 1)

        output[trait] = {
            "trait": trait,
            "available": True,
            "n_cells": len(values),
            "n_persons": len(person_means),
            "common_direction_shift": {
                "estimate": _finite(estimate), "se": _finite(se), "ci": common_ci,
                "p": common_p,
            },
            "group_heterogeneity_p": _finite(group_p),
            "wave_heterogeneity_p": _finite(wave_p),
            "cell_heterogeneity_p": _finite(cell_p),
            "note": ("common_direction_shift 是反向题相对正向题的平均平移，只平移 C 的"
                     "绝对水平。三个异质性 p 值来自置换检验（组别在学员间置换，轮次在"
                     "学员内置换），显著时平衡内容差值不能直接解释为体验变化。"),
        }
    # 同一检验族跨 trait 做 Holm 校正
    for key in ("group_heterogeneity_p", "wave_heterogeneity_p", "cell_heterogeneity_p"):
        names = [t for t in traits if output[t].get("available")]
        adjusted = _holm([output[t][key] for t in names])
        for name, value in zip(names, adjusted):
            output[name][key.replace("_p", "_holm")] = value
    return output


# ---------------------------------------------------------------------------
# 候选模型：比较内容与作答方式的解释力
# ---------------------------------------------------------------------------
#
# 观测单位是 (学员, 轮次, 题)。因变量是内容方向归一化后再减去训练折题目位置的
# 残差 y（题目位置由训练折单独估计，避免使用被预测学员）。候选设计只用登记字段：
#
#   item      截距；组别/轮次/方向都无差异（最低基线）
#   direct    格×轮次（组别组合与轮次）；直接比较观察分数，不区分正反向
#   content   格×轮次 + 方向；方向负担只造成跨单元格的共同平移
#   style     格×轮次 + 方向 + 方向×格×轮次；方向负担随组别与轮次变化
#   vignette  用训练折的固定情境均值（归一化后去中心）作为内容预测；仅在情境
#             通过顺序与收敛检验时参与
#
# 模型只通过整位学员的嵌套折外均方误差比较：内层在训练折内选型，外层给出样本外
# 误差；不使用样本内 R² 或拟合优度。整位 holdout 学员在选型结束后只做一次检查。

def _observed_keys(evidence, persons, trait, group_by):
    """训练折中出现过、且该 trait 有作答的格（组别组合×轮次）。"""
    return sorted({_cell_key(evidence, p, w, group_by)
                   for p in persons for w in evidence.waves
                   if any(evidence.items[n].trait == trait
                          for n in evidence.answers.get((p, w), {}))})


def _design_matrix(evidence, persons, trait, locations, group_by, model, keys):
    """返回逐观测设计矩阵、因变量、聚类标签与预测所需的元数据。"""
    index = {key: n for n, key in enumerate(keys)}
    columns = ["intercept"]
    if model in ("direct", "content", "style"):
        columns += [f"cell:{key}" for key in keys[1:]]
    if model in ("content", "style"):
        columns.append("reverse")
    if model == "style":
        columns += [f"reverse:cell:{key}" for key in keys[1:]]
    position = {name: n for n, name in enumerate(columns)}
    rows, outcomes, clusters, meta = [], [], [], []
    for person in persons:
        for wave in evidence.waves:
            answered = evidence.answers.get((person, wave), {})
            for name, raw in sorted(answered.items()):
                item = evidence.items[name]
                if item.trait != trait or name not in locations:
                    continue
                key = _cell_key(evidence, person, wave, group_by)
                if key not in index:
                    continue
                row = np.zeros(len(columns))
                row[position["intercept"]] = 1.0
                if model in ("direct", "content", "style") and index[key] > 0:
                    row[position[f"cell:{key}"]] = 1.0
                if model in ("content", "style"):
                    row[position["reverse"]] = 1.0 if item.reverse else 0.0
                if model == "style" and index[key] > 0:
                    row[position[f"reverse:cell:{key}"]] = (
                        1.0 if item.reverse else 0.0)
                rows.append(row)
                outcomes.append(_content_value(item, raw) - locations[name])
                clusters.append(person)
                meta.append((key, item.reverse, name))
    return (np.array(rows) if rows else np.zeros((0, len(columns))),
            np.array(outcomes), clusters, meta, columns)


def _vignette_cell_values(evidence, persons, trait, group_by):
    block = [name for name, spec in evidence.vignettes.items() if spec["trait"] == trait]
    values = defaultdict(list)
    for person in persons:
        for wave in evidence.waves:
            answers = evidence.vignette_answers.get((person, wave), {})
            present = [v for v in block if v in answers]
            if present:
                key = _cell_key(evidence, person, wave, group_by)
                values[key].append(float(np.mean([_normalize(answers[v],
                                                             Item(v, trait, False, 1.0, 5.0))
                                                  for v in present])))
    return {key: float(np.mean(v)) for key, v in values.items() if v}


def _fit_predict(evidence, train_persons, test_persons, trait, group_by, model, seed):
    train_locations = item_locations(evidence, train_persons)
    keys = _observed_keys(evidence, train_persons, trait, group_by)
    if not keys:
        return None
    if model == "vignette":
        centers = _vignette_cell_values(evidence, train_persons, trait, group_by)
        if not centers:
            return None
        grand = _mean(centers.values())
        test_keys = {_cell_key(evidence, p, w, group_by)
                     for p in test_persons for w in evidence.waves}
        if not any(key in centers for key in test_keys):
            return None
        squared, count = [], 0
        for person in test_persons:
            for wave in evidence.waves:
                key = _cell_key(evidence, person, wave, group_by)
                if key not in centers:
                    continue
                predicted = centers[key] - grand
                for name, raw in sorted(evidence.answers.get((person, wave), {}).items()):
                    item = evidence.items[name]
                    if item.trait != trait or name not in train_locations:
                        continue
                    observed = _content_value(item, raw) - train_locations[name]
                    squared.append((observed - predicted) ** 2)
                    count += 1
        if not squared:
            return None
        return {"mse": float(np.mean(squared)), "n": count}

    design, outcome, _, _, columns = _design_matrix(
        evidence, train_persons, trait, train_locations, group_by, model, keys)
    if len(outcome) == 0:
        return None
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    test_design, test_outcome, _, _, _ = _design_matrix(
        evidence, test_persons, trait, train_locations, group_by, model, keys)
    if len(test_outcome) == 0:
        return None
    prediction = test_design @ beta
    return {"mse": float(np.mean((prediction - test_outcome) ** 2)), "n": len(test_outcome)}


def model_comparison(evidence, fit_persons, trait, config, vignette_supported):
    models = [name for name in CANDIDATE_ORDER
              if name != "vignette" or vignette_supported]
    outer = {name: [] for name in models}
    inner_selections = []
    for outer_fold in range(config["folds"]):
        test = [p for p in fit_persons
                if _fold(p, config["seed"], "outer", config["folds"]) == outer_fold]
        train = [p for p in fit_persons if p not in set(test)]
        if not train or not test:
            continue
        inner = {name: [] for name in models}
        for inner_fold in range(config["folds"]):
            validation = [p for p in train
                          if _fold(p, config["seed"], "inner", config["folds"]) == inner_fold]
            training = [p for p in train if p not in set(validation)]
            if not training or not validation:
                continue
            for name in models:
                result = _fit_predict(evidence, training, validation, trait,
                                      config["group_by"], name, config["seed"])
                if result is not None:
                    inner[name].append(result["mse"])
        inner_means = {name: _mean(values) for name, values in inner.items()}
        usable = {name: value for name, value in inner_means.items() if value is not None}
        chosen = None
        if usable:
            best = min(usable.values())
            chosen = next(name for name in models
                          if usable.get(name) is not None and usable[name] <= best + 1e-9)
        inner_selections.append(chosen)
        for name in models:
            result = _fit_predict(evidence, train, test, trait, config["group_by"],
                                  name, config["seed"])
            if result is not None:
                outer[name].append(result["mse"])
    summary = {}
    for name in models:
        summary[name] = {
            "description": CANDIDATE_DESCRIPTION[name],
            "outer_fold_mse": _mean(outer[name]),
            "outer_fold_mse_values": [_finite(v) for v in outer[name]],
            "outer_folds": len(outer[name]),
        }
    counts = {name: sum(1 for choice in inner_selections if choice == name)
              for name in models}
    selected = None
    if any(counts.values()):
        selected = max(models, key=lambda name: (counts[name], -CANDIDATE_ORDER.index(name)))
    return {
        "trait": trait,
        "selected": selected,
        "models": summary,
        "inner_selected_counts": counts,
        "inner_selection_folds": len(inner_selections),
        "vignette_included": bool(vignette_supported),
        "note": ("模型比较使用整位学员的嵌套折外均方误差；内层选型，外层评估。观察为"
                 "内容方向归一化并减去训练折题目位置的残差。折外误差小不等于组别可"
                 "校准，必须结合信度与不变性诊断；item 被选中表示组别/轮次差异没有"
                 "超出噪声的可重复证据。"),
    }


def holdout_check(evidence, fit_persons, holdout_persons, trait, config, selected_model):
    if not holdout_persons or selected_model is None:
        return {"trait": trait, "available": False, "selected_model": selected_model,
                "holdout_mse": None, "holdout_observations": None,
                "n_holdout_persons": len(holdout_persons),
                "reason": "没有选中的模型或没有留出学员"}
    result = _fit_predict(evidence, fit_persons, holdout_persons, trait,
                          config["group_by"], selected_model, config["seed"])
    return {
        "trait": trait,
        "available": result is not None,
        "selected_model": selected_model,
        "holdout_mse": result["mse"] if result else None,
        "holdout_observations": result["n"] if result else None,
        "n_holdout_persons": len(holdout_persons),
        "note": ("留出学员只用于此一次检查；其记录未参与题目位置、信度、不变性检验或"
                 "模型选择。"),
    }


# ---------------------------------------------------------------------------
# 可比性判定与结论
# ---------------------------------------------------------------------------

def _comparability(evidence, trait, config, reliability_block, style_block,
                   invariance_block, direction_block, vignette_block):
    """把各项检验汇总为可比性约束清单；任一关键前提失败即返回不支持。"""
    constraints = []

    def add(name, status, detail):
        constraints.append({"constraint": name, "status": status, "detail": detail})

    reliability_ok = reliability_block["status"] == "adequate"
    add("content_reliability", "pass" if reliability_ok else "fail",
        "内容分数具有可重复信度" if reliability_ok
        else "内容分数信度不足，无法把变化与噪声区分")

    group_ok = not invariance_block["failed_group_items"]
    add("item_group_invariance", "pass" if group_ok else "fail",
        "各题作答分布不随组别变化" if group_ok
        else "题目分布随组别变化：" + ", ".join(invariance_block["failed_group_items"]))
    wave_ok = not invariance_block["failed_wave_items"]
    add("item_wave_invariance", "pass" if wave_ok else "fail",
        "各题作答分布不随轮次变化" if wave_ok
        else "题目分布随轮次变化：" + ", ".join(invariance_block["failed_wave_items"]))
    missing_ok = not invariance_block["differential_missing_items"]
    add("missingness_balance", "pass" if missing_ok else "fail",
        "缺失率在组别间可比" if missing_ok
        else "缺失率在组别间差异过大：" +
             ", ".join(invariance_block["differential_missing_items"]))

    direction_ok = True
    detail = "未发现方向负担随组别或轮次变化的证据"
    if direction_block.get("available"):
        flagged = [name for name in ("group_heterogeneity_holm", "wave_heterogeneity_holm",
                                     "cell_heterogeneity_holm")
                   if direction_block.get(name) is not None
                   and direction_block[name] < config["alpha"]]
        if flagged:
            direction_ok = False
            detail = ("方向负担随组别或轮次变化（" + ", ".join(
                f"{name}={direction_block[name]:.3g}" for name in flagged)
                + "），平衡内容差值不能直接解释为体验变化")
    add("direction_heterogeneity", "pass" if direction_ok else "fail", detail)

    style_detail = "作答方式在各组×轮的水平已报告；未作为校准量使用"
    if style_block["style_icc"] is not None and style_block["style_icc"] < 0:
        style_detail = "作答方式 ICC 为负，方向差分缺乏重复性"
    add("style_repeatability", "info", style_detail)

    levels = style_block.get("levels") or []
    means = [level["mean_style"] for level in levels if level.get("mean_style") is not None]
    if len(means) >= 2:
        spread = max(means) - min(means)
        material = spread > config["style_tolerance"]
        add("style_level_spread", "warn" if material else "pass",
            f"各组×轮的平均作答方式极差为 {spread:.3f}，"
            + (f"超过登记容差 {config['style_tolerance']:.2f}；即使未达到统计显著，"
               "也应把该校准幅度写入报告" if material
               else f"在登记容差 {config['style_tolerance']:.2f} 之内；"
                    "作答方式差异不足以支撑按组校准"))

    vignette_detail = ("情境未通过检验，未作为锚" if vignette_block["status"] != "supported"
                       else "情境通过检验，仅用于三角验证与敏感性分析")
    add("vignette_anchor", "info", vignette_detail)

    supported = all(c["status"] != "fail" for c in constraints)
    return {
        "trait": trait,
        "comparable": bool(supported),
        "constraints": constraints,
        "failed_constraints": [c["constraint"] for c in constraints if c["status"] == "fail"],
        "allows_group_comparison": bool(supported),
        "allows_wave_change_claims": bool(supported),
        "vignette_status": vignette_block["status"],
    }


def _group_wave_verdicts(changes_block, reliability_block, comparability_block, config):
    """逐组×轮、逐变化量给出“是否仍有可比依据”。"""
    verdicts = []
    for entry in changes_block:
        for group in [{"level": {}, **entry["overall"]}] + entry["groups"]:
            available = (comparability_block["comparable"]
                         and group.get("ci") is not None)
            excludes_zero = (available and group["ci"] is not None
                             and (group["ci"][0] > 0 or group["ci"][1] < 0))
            verdicts.append({
                "waves": entry["waves"],
                "measure": entry["measure"],
                "level": group["level"],
                "change": group.get("change"),
                "ci": group.get("ci"),
                "n_pairs": group.get("n_pairs"),
                "supported": bool(excludes_zero),
                "reason": ("信度与识别约束通过，且区间不含 0" if excludes_zero
                           else ("识别约束未通过，变化量只作描述" if not available
                                 else "区间包含 0，不能宣称有变化")),
            })
    return verdicts


# ---------------------------------------------------------------------------
# 敏感性分析：结论依赖哪些题、哪些锚、哪些假设
# ---------------------------------------------------------------------------

def _subset_cells(evidence, persons, traits, locations, trait, items_excluded,
                  directions, drop_group):
    keep = {name: item for name, item in evidence.items.items()
            if item.trait in traits and name not in items_excluded
            and (directions is None or item.reverse in directions)}
    subset = Evidence(people=evidence.people, items=keep, answers=evidence.answers,
                      vignettes=evidence.vignettes,
                      vignette_answers=evidence.vignette_answers,
                      waves=evidence.waves, persons=evidence.persons,
                      fit_persons=evidence.fit_persons,
                      holdout_persons=evidence.holdout_persons,
                      group_levels=evidence.group_levels)
    used = [p for p in persons
            if drop_group is None or evidence.people[p][drop_group[0]] != drop_group[1]]
    return cell_indices(subset, used, locations, traits)


def sensitivity_analysis(evidence, fit_persons, traits, config, vignette_blocks):
    locations = item_locations(evidence, fit_persons)
    base_cells = cell_indices(evidence, fit_persons, locations, traits)
    group_by = config["group_by"]
    pairs = config["wave_pairs"]
    rng = np.random.default_rng([config["seed"], 62])
    output = {}
    for trait in traits:
        base = {}
        if pairs:
            for field, measure in (("C", "content"), ("S", "style")):
                base[measure] = _paired(evidence, base_cells, fit_persons, trait,
                                        pairs[0], field)
        base_values = base.get("content", [])
        base_mean = _mean(base_values)
        base_ci = _ci(_bootstrap_values(rng, base_values, config["bootstrap"], _mean))
        # 只有基准变化本身可与 0 区分时，单题/单方向导致的符号翻转才构成脆弱。
        base_decisive = bool(base_ci and (base_ci[0] > 0 or base_ci[1] < 0))
        scenarios = []

        def record(label, removed_items, directions, drop_group, kind,
                   uses_vignette=False, assumption=None):
            cells = _subset_cells(evidence, fit_persons, [trait], locations, trait,
                                  removed_items, directions, drop_group)
            values = _paired(evidence, cells, fit_persons, trait, pairs[0], "C",
                             require_balanced=(directions is None)) \
                if pairs else []
            reference = base_mean
            current = _mean(values)
            scenarios.append({
                "scenario": label,
                "kind": kind,
                "removed_items": sorted(removed_items),
                "directions": None if directions is None else
                ([ "forward" if d is False else "reverse" for d in directions]),
                "dropped_group": None if drop_group is None
                else {drop_group[0]: drop_group[1]},
                "uses_vignette_anchor": uses_vignette,
                "assumption": assumption,
                "change": current,
                "ci": _ci(_bootstrap_values(rng, values, config["bootstrap"], _mean))
                if len(values) >= 3 else None,
                "n_pairs": len(values),
                "same_sign_as_base": bool(reference is not None and current is not None
                                          and (reference == 0
                                               or (reference > 0) == (current > 0))),
                # 基准可区分且该情景翻转符号，才判定结论依赖该情景。
                "changes_conclusion": bool(base_decisive and current is not None
                                           and (current == 0 or (current > 0) !=
                                                (base_mean > 0))),
            })

        record("base:balanced", set(), None, None, "base")
        items = sorted(name for name, item in evidence.items.items() if item.trait == trait)
        for name in items:
            record(f"exclude_item:{name}", {name}, None, None, "item")
        record("forward_only", set(), {False}, None, "direction",
               assumption="只用正向题；默认赞同无法与内容分离")
        record("reverse_only", set(), {True}, None, "direction",
               assumption="只用反向题；反向题理解负担直接进入内容")
        for field in group_by:
            for level in evidence.group_levels[field]:
                record(f"exclude_group:{field}={level}", set(), None, (field, level),
                       "group")
        output[trait] = {
            "trait": trait,
            "wave_pair": pairs[0] if pairs else None,
            "base_change": {
                "mean": base_mean, "sd": _sd(base_values), "ci": base_ci,
                "n_pairs": len(base_values), "distinguishable_from_zero": base_decisive,
            },
            "scenarios": scenarios,
            "vignette": {
                "status": vignette_blocks[trait]["status"],
                "used_as_anchor": vignette_blocks[trait]["status"] == "supported",
                "note": ("情境仅在通过顺序与收敛检验时参与模型比较；无论是否通过，"
                         "都不用于把组别校准到同一分布。"),
            },
            "depends_on_items": sorted({item for scenario in scenarios
                                        if scenario["kind"] == "item"
                                        and scenario["changes_conclusion"]
                                        for item in scenario["removed_items"]}),
            "assumption_dependent_scenarios": [
                scenario["scenario"] for scenario in scenarios
                if scenario["kind"] == "direction" and scenario["changes_conclusion"]],
        }
    return output


def _conclusion(traits, comparability, reliability_blocks, style_blocks,
                vignette_blocks, sensitivity, direction_tests):
    conclusions = {}
    for trait in traits:
        block = comparability[trait]
        failed = block["failed_constraints"]
        if block["comparable"]:
            verdict = "qualified"
            statement = ("该 trait 的平衡内容差值在信度与识别约束通过的前提下可作为有条件"
                         "的描述性对比；报告时须同时给出区间、敏感性结果与识别假设。")
        else:
            verdict = "not_supported"
            statement = ("该 trait 目前没有可比依据，不应把“更有信心、归属感更强”等"
                         "变化写进项目总结。")
        fragile = sensitivity[trait].get("depends_on_items", [])
        conclusions[trait] = {
            "verdict": verdict,
            "statement": statement,
            "failed_constraints": failed,
            "reliability_status": reliability_blocks[trait]["status"],
            "stable_style_trait": style_blocks[trait]["stable_style_trait"],
            "vignette_status": vignette_blocks[trait]["status"],
            "direction_heterogeneous": bool(
                direction_tests[trait].get("available") and any(
                    direction_tests[trait].get(name) is not None
                    and direction_tests[trait][name] < 0.05
                    for name in ("group_heterogeneity_holm", "wave_heterogeneity_holm",
                                 "cell_heterogeneity_holm"))),
            "fragile_items": fragile,
            "dependencies": {
                "items": ("去掉任一题后变化量符号改变：" + ", ".join(fragile)
                          if fragile else "变化量符号不依赖单题"),
                "anchors": ("固定情境未通过检验，未作为锚" if
                            vignette_blocks[trait]["status"] != "supported"
                            else "固定情境通过检验，仅用于三角验证"),
                "constraints": failed or ["全部识别约束通过"],
            },
            "next_evidence": ("需要先提升内容信度（题目间共同变异）并确认反向题理解一致，"
                              "再谈跨校区或跨轮比较；当前数据不足以支撑项目总结中的"
                              "变化表述。") if not block["comparable"] else
                             "继续监测方向负担是否随组别/轮次漂移。",
        }
    return conclusions


def response_style_analysis(tables, parameters):
    config = parse_parameters(tables, parameters)
    evidence = build_evidence(tables, config)
    traits = config["traits"]
    locations = item_locations(evidence, evidence.fit_persons)
    cells = cell_indices(evidence, evidence.fit_persons, locations, traits)

    reliability_blocks = {trait: reliability(evidence, evidence.fit_persons, trait, config)
                          for trait in traits}
    style_blocks = {trait: style_structure(evidence, cells, evidence.fit_persons, trait, config)
                    for trait in traits}
    style_level_blocks = {trait: style_levels(evidence, cells, evidence.fit_persons,
                                              trait, config) for trait in traits}
    for trait in traits:
        style_blocks[trait]["levels"] = style_level_blocks[trait]
    invariance = dif_tests(evidence, cells, evidence.fit_persons, traits, config)
    vignette_blocks = vignette_diagnostics(evidence, evidence.fit_persons, traits, config)
    direction_blocks = direction_tests(evidence, cells, evidence.fit_persons, traits, config)
    comparability = {
        trait: _comparability(evidence, trait, config, reliability_blocks[trait],
                              style_blocks[trait], invariance[trait],
                              direction_blocks[trait], vignette_blocks[trait])
        for trait in traits}
    change_blocks = changes(evidence, cells, evidence.fit_persons, traits, config)
    verdicts = {trait: _group_wave_verdicts(change_blocks[trait], reliability_blocks[trait],
                                            comparability[trait], config)
                for trait in traits}
    models = {trait: model_comparison(evidence, evidence.fit_persons, trait, config,
                                      vignette_blocks[trait]["status"] == "supported")
              for trait in traits}
    holdout = {trait: holdout_check(evidence, evidence.fit_persons,
                                    evidence.holdout_persons, trait, config,
                                    models[trait]["selected"]) for trait in traits}
    sensitivity = sensitivity_analysis(evidence, evidence.fit_persons, traits, config,
                                       vignette_blocks)
    conclusions = _conclusion(traits, comparability, reliability_blocks, style_blocks,
                              vignette_blocks, sensitivity, direction_blocks)

    return {
        "version": VERSION,
        "product": "学习体验追踪",
        "design": {
            "group_by": config["group_by"],
            "traits": traits,
            "wave_pairs": config["wave_pairs"],
            "holdout_fraction": config["holdout_fraction"],
            "folds": config["folds"],
            "alpha": config["alpha"],
            "reliability_threshold": config["reliability_threshold"],
            "n_fit_persons": len(evidence.fit_persons),
            "n_holdout_persons": len(evidence.holdout_persons),
            "n_waves": len(evidence.waves),
            "fit_persons": evidence.fit_persons,
            "holdout_persons": evidence.holdout_persons,
            "identification": [
                "正反向题等权时 C=(f+r)/2 消除共同的加性默认赞同；比较 C 的差值不要求"
                "把各校区校准到同一分布。",
                "S=(f-r)/2 是方向偏差；只有它与组别/轮次缠结时，C 的差值才失去可比性。",
                "反向题若携带随组别或轮次变化的额外理解负担，C 与内容变化在观测上"
                "部分等价，只能由不变性检验与方向×组/轮检验间接核查。",
                "固定情境必须先通过顺序与收敛检验，才可作为效标；不用于强行校准。",
                "1–5 为有序等级；均值与差值只作描述。",
                "所有拟合、诊断与选型只用 fit 学员；holdout 学员仅用于一次样本外检查。",
            ],
        },
        "item_locations": {name: _finite(value)
                           for name, value in sorted(locations.items())},
        "reliability": reliability_blocks,
        "response_style": style_blocks,
        "response_style_levels": style_level_blocks,
        "invariance": invariance,
        "direction_tests": direction_blocks,
        "vignettes": vignette_blocks,
        "comparability": comparability,
        "changes": change_blocks,
        "change_verdicts": verdicts,
        "model_selection": models,
        "holdout": holdout,
        "sensitivity": sensitivity,
        "conclusions": conclusions,
        "limitations": LIMITATIONS,
    }
