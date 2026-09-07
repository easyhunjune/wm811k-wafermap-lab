"""Build the verdict sentences the aggregate reports print, from the deltas themselves.

The reports used to carry these judgements as literal strings, so rerunning an
aggregation with more seeds refreshed every table while the conclusions underneath
kept the numbers and denominators of the first run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaVerdict:
    """A paired per-seed difference, summarised for a judgement."""

    mean: float
    std: float
    positive: int
    negative: int
    total: int

    @property
    def moved(self) -> bool:
        """At least one seed changed at all."""
        return self.positive + self.negative > 0

    @property
    def unanimous(self) -> bool:
        """Every seed moved the way the mean did."""
        return self.moved and self.agreeing == self.total

    @property
    def exceeds_spread(self) -> bool:
        return abs(self.mean) > self.std

    @property
    def consistent(self) -> bool:
        return self.unanimous and self.exceeds_spread

    @property
    def direction(self) -> str:
        if self.mean == 0:
            return "변화 없음" if not self.moved else "상쇄"
        return "하락" if self.mean < 0 else "상승"

    @property
    def agreeing(self) -> int:
        """Seeds that moved the way the mean did."""
        return self.negative if self.mean < 0 else self.positive

    def describe(self, name: str) -> str:
        return f"{name} {self.mean:+.4f} ({self.direction} {self.agreeing}/{self.total} seed)"


def summarise_delta(deltas) -> DeltaVerdict:
    values = np.asarray(deltas, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("A paired verdict needs at least two seeds.")
    return DeltaVerdict(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        positive=int((values > 0).sum()),
        negative=int((values < 0).sum()),
        total=int(values.size),
    )


def macro_f1_verdict(comparison: str, deltas) -> str:
    """Judge a paired macro F1 difference against the seed-to-seed spread."""
    return metric_verdict(comparison, "macro F1", deltas)


def metric_verdict(comparison: str, metric: str, deltas) -> str:
    """Judge a paired difference in any per-seed metric against the seed-to-seed spread."""
    verdict = summarise_delta(deltas)
    if not verdict.moved:
        return (
            f"- {comparison} {metric}: 모든 seed에서 차이가 0이다. "
            f"({verdict.total} seed)"
        )
    if verdict.consistent:
        reading = "평균 차이가 seed 간 표준편차를 넘고 방향도 모든 seed에서 일치한다"
        # A unanimous, spread-exceeding difference is a finding in either direction:
        # the left side is better, or it is worse. Only the first is an improvement.
        claim = (
            "고정 분할에서의 개선으로 판정한다"
            if verdict.mean > 0
            else "고정 분할에서의 하락으로 판정한다"
        )
    elif verdict.unanimous:
        reading = "방향은 모든 seed에서 일치하나 평균 차이가 seed 간 표준편차를 넘지 않는다"
        claim = "일관된 차이로 판정하지 않는다"
    elif verdict.exceeds_spread:
        reading = "평균 차이는 seed 간 표준편차를 넘지만 방향이 seed마다 갈린다"
        claim = "일관된 차이로 판정하지 않는다"
    else:
        reading = "평균 차이가 seed 간 표준편차를 넘지 않고 방향도 seed마다 갈린다"
        claim = "일관된 차이로 판정하지 않는다"
    return (
        f"- {comparison} {metric}: 평균 {verdict.mean:+.4f}, "
        f"seed 간 표준편차 {verdict.std:.4f}, "
        f"상승 {verdict.positive}/{verdict.total} seed. {reading}. {claim}."
    )


def class_tradeoff_verdict(label: str, *, recall_deltas, precision_deltas) -> str:
    """Report whether a class traded precision for recall, and how consistently."""
    recall = summarise_delta(recall_deltas)
    precision = summarise_delta(precision_deltas)
    # Counts alone can point the opposite way to the mean: a majority of seeds may nudge
    # recall up while one large drop pulls the average down. Both the direction label and
    # the pattern test therefore follow the mean, never the raw count.
    traded = recall.mean > 0 and precision.mean < 0
    # Count the seeds where the trade-off actually happened together. Two separate
    # majorities can overlap in a minority of seeds, which is not the same finding.
    both = int(
        (
            (np.asarray(recall_deltas, dtype=float) > 0)
            & (np.asarray(precision_deltas, dtype=float) < 0)
        ).sum()
    )
    if traded and both == recall.total:
        reading = "재현율 상승과 정밀도 하락이 모든 seed에서 반복됐다"
        short = [
            name
            for name, verdict in (("재현율", recall), ("정밀도", precision))
            if not verdict.exceeds_spread
        ]
        if short:
            reading += f", 다만 {'·'.join(short)}의 평균 차이는 seed 간 표준편차를 넘지 않는다"
    elif traded and both * 2 > recall.total:
        reading = (
            f"재현율 상승과 정밀도 하락이 {both}/{recall.total} seed에서 함께 나타났으나 "
            "모든 seed에서 일치하지는 않는다"
        )
    else:
        # Say only that the trade-off pattern is absent. Whether each metric moved
        # consistently on its own is a separate question, answered next.
        reading = "재현율 상승과 정밀도 하락의 절충 패턴은 확인되지 않는다"
        unanimous = [
            name
            for name, verdict in (("재현율", recall), ("정밀도", precision))
            if verdict.unanimous
        ]
        if unanimous:
            reading += f". 다만 {'·'.join(unanimous)}는 모든 seed에서 같은 방향이다"
    return f"- {label}: {recall.describe('재현율')}, {precision.describe('정밀도')}. {reading}."


def leading_candidate_line(macro_f1_by_experiment, candidates: tuple[str, ...]) -> str:
    """Name the candidate with the highest mean macro F1, and say how it is separated."""
    if len(candidates) < 2:
        raise ValueError("Naming a leading candidate needs at least two candidates.")
    if any(len(macro_f1_by_experiment[name]) < 2 for name in candidates):
        raise ValueError("Comparing candidates needs at least two seeds each.")
    means = {name: float(macro_f1_by_experiment[name].mean()) for name in candidates}
    stds = {name: float(macro_f1_by_experiment[name].std(ddof=1)) for name in candidates}
    ranked = sorted(candidates, key=lambda name: means[name], reverse=True)
    leader, runner_up = ranked[0], ranked[1]
    if means[leader] == means[runner_up]:
        return (
            f"- {leader}와 {runner_up}의 macro F1 평균이 {means[leader]:.4f}로 같다. "
            "이 지표만으로는 선도 후보를 가릴 수 없다."
        )
    if stds[leader] < stds[runner_up]:
        spread = "표준편차도 더 작다"
    elif stds[leader] > stds[runner_up]:
        spread = "표준편차는 더 크다"
    else:
        spread = "표준편차는 같다"
    # Only macro F1 enters this function, so the sentence claims nothing beyond it.
    return (
        f"- {leader}는 {runner_up}보다 macro F1 평균이 높고({means[leader]:.4f} vs "
        f"{means[runner_up]:.4f}) {spread}"
        f"({stds[leader]:.4f} vs {stds[runner_up]:.4f}). "
        f"사전 주 지표인 macro F1을 기준으로 {leader}를 선도 후보로 둔다."
    )


def spread_line(metric: str, before, after, *, before_name: str, after_name: str) -> str:
    """Compare run-to-run spread, which a paired delta on its own does not show."""
    before_std = float(np.asarray(before, dtype=float).std(ddof=1))
    after_std = float(np.asarray(after, dtype=float).std(ddof=1))
    # Identical values do not always give exactly 0.0 back from std(), so compare against
    # a tolerance rather than zero; the ratio below is meaningless either way.
    if before_std < 1e-12 or after_std < 1e-12:
        change = "비교 불가(한쪽 표준편차가 0)"
    elif after_std > before_std:
        ratio = after_std / before_std
        # A ratio that rounds to 1.0x is not a change worth naming a direction for.
        change = f"{ratio:.1f}배로 커졌다" if ratio >= 1.05 else "사실상 같다"
    else:
        ratio = before_std / after_std
        change = f"{ratio:.1f}배로 작아졌다" if ratio >= 1.05 else "사실상 같다"
    return (
        f"- {metric} seed 간 표준편차: `{before_name}` {before_std:.4f} → "
        f"`{after_name}` {after_std:.4f}. 실행 간 변동이 {change}."
    )


def count_verdict(comparison: str, deltas) -> str:
    """Judge a paired count difference, such as `none` false positives."""
    verdict = summarise_delta(deltas)
    if not verdict.moved:
        return f"- {comparison}: 모든 seed에서 차이가 0이다. ({verdict.total} seed)"
    # Report the seed count that agrees with the mean's direction, so an increase is not
    # described by how many seeds went the other way.
    direction = "감소" if verdict.mean < 0 else "증가"
    consistency = (
        "모든 seed에서 같은 방향이다" if verdict.unanimous else "방향이 seed마다 갈린다"
    )
    return (
        f"- {comparison}: 평균 {abs(verdict.mean):.1f}건 {direction} "
        f"(± {verdict.std:.1f}), {direction} {verdict.agreeing}/{verdict.total} seed. "
        f"{consistency}."
    )
