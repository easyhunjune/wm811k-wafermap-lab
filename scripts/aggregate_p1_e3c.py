from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_p1_e3b import count_delta_text, delta_text
from aggregate_seeds import DEFAULT_SEEDS, FOCUS_LABELS, describe, load_runs, sha256

from wm811k.verdicts import (
    class_tradeoff_verdict,
    leading_candidate_line,
    macro_f1_verdict,
)

EXPERIMENTS = ("E2", "E3", "E3b", "E3c")
BASELINES = ("E2", "E3", "E3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the P1 E3c comparison.")
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "repeat"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "P1_E3C.md"
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "p1_e3c_macro_f1_box.png",
    )
    return parser.parse_args()


def save_figure(summary, path: Path) -> None:
    values = [
        summary.loc[summary["experiment"] == experiment, "macro_f1"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot(values, tick_labels=EXPERIMENTS, showmeans=True)
    for index, run_values in enumerate(values, start=1):
        axis.scatter(np.full(len(run_values), index), run_values, alpha=0.7, zorder=3)
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Validation macro F1")
    axis.set_title("P1 E3c: capped inverse-frequency weighting")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(summary, classes, *, report: Path, figure: Path, split_hash: str) -> None:
    try:
        figure_link = figure.relative_to(report.parent).as_posix()
    except ValueError:
        figure_link = figure.as_posix()
    macro = summary.pivot(index="seed", columns="experiment", values="macro_f1")
    balanced = summary.pivot(
        index="seed", columns="experiment", values="balanced_accuracy"
    )
    none_fp = summary.pivot(
        index="seed", columns="experiment", values="none_false_positives"
    )
    precision = classes.pivot(
        index=["seed", "label"], columns="experiment", values="precision"
    )
    recall = classes.pivot(
        index=["seed", "label"], columns="experiment", values="recall"
    )
    lines = [
        "# P1 E3c: 역빈도 가중치 상한",
        "",
        "## 사전 등록",
        "",
        "- 독립 변수: 역빈도 클래스 가중치 상한 `max_min_ratio=32.0`",
        "- 선정 근거: E3b의 약 32.23:1 동적 범위와 맞춘 사전 고정값",
        "- E3b는 전체 가중치 분포를 제곱근으로 완화하고 E3c는 희소 클래스 꼬리만 제한한다.",
        "- 학습 seed: 42, 52, 62, 72, 82",
        f"- 분할 SHA-256: `{split_hash}`",
        "- 테스트 데이터 사용 여부: 사용하지 않음",
        "",
        "## 검증 macro F1",
        "",
        "| 실험 | 평균 ± 표준편차 | 최소–최대 |",
        "|---|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        stats = describe(summary.loc[summary["experiment"] == experiment, "macro_f1"])
        lines.append(
            f"| {experiment} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
            f"{stats['min']:.4f}–{stats['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"![P1 E3c macro F1 box plot]({figure_link})",
            "",
            "## 대응 차이",
            "",
        ]
    )
    for baseline in BASELINES:
        lines.extend(
            [
                f"- E3c−{baseline} macro F1: {delta_text(macro['E3c'] - macro[baseline])}",
                (
                    f"- E3c−{baseline} balanced accuracy: "
                    f"{delta_text(balanced['E3c'] - balanced[baseline])}"
                ),
                (
                    f"- E3c−{baseline} `none` false positive: "
                    f"{count_delta_text(none_fp['E3c'] - none_fp[baseline])}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 클래스별 정밀도·재현율 변화",
            "",
            "| 비교 | 클래스 | 정밀도 변화 | 재현율 변화 |",
            "|---|---|---:|---:|",
        ]
    )
    for baseline in BASELINES:
        for label in FOCUS_LABELS:
            p = precision.xs(label, level="label")
            r = recall.xs(label, level="label")
            lines.append(
                f"| E3c−{baseline} | {label} | "
                f"{delta_text(p['E3c'] - p[baseline])} | "
                f"{delta_text(r['E3c'] - r[baseline])} |"
            )
    lines.extend(
        [
            "",
            "## 판정 규칙",
            "",
            "- E3c가 E3b보다 macro F1을 유지하면서 Loc·Edge-Loc 재현율을 개선하는지 확인한다.",
            "- 정밀도 또는 macro F1이 크게 하락하면 재현율 상승만으로 개선 판정하지 않는다.",
            "- 평균 차이가 seed 변동보다 작거나 방향이 일관되지 않으면 우월성을 주장하지 않는다.",
            "",
            "## 결론",
            "",
            macro_f1_verdict("E3c−E3", macro["E3c"] - macro["E3"]),
            macro_f1_verdict("E3c−E3b", macro["E3c"] - macro["E3b"]),
            class_tradeoff_verdict(
                "Edge-Loc (E3c−E3b)",
                recall_deltas=(
                    recall.xs("Edge-Loc", level="label")["E3c"]
                    - recall.xs("Edge-Loc", level="label")["E3b"]
                ),
                precision_deltas=(
                    precision.xs("Edge-Loc", level="label")["E3c"]
                    - precision.xs("Edge-Loc", level="label")["E3b"]
                ),
            ),
            leading_candidate_line(macro, ("E3b", "E3c")),
            "",
            "## P1 정지 결정",
            "",
            (
                "P1의 목표였던 강한 역빈도 가중치 완화 방법을 E3b와 E3c로 비교했고 "
                "E3b가 더 안정적인 절충임을 확인했다. E4~E6은 선택적 확장 실험으로 "
                "남기며, 검증 데이터에 대한 반복 조정을 피하기 위해 현재 P1을 종료한다. "
                "테스트 데이터는 갱신하지 않는다."
            ),
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    split_hash = sha256(args.splits)
    summary, classes = load_runs(
        args.input_dir,
        seeds=DEFAULT_SEEDS,
        expected_split_hash=split_hash,
        experiments=EXPERIMENTS,
    )
    summary_path = args.input_dir / "p1_e3c_summary.csv"
    class_path = args.input_dir / "p1_e3c_class_metrics.csv"
    summary.to_csv(summary_path, index=False)
    classes.to_csv(class_path, index=False)
    save_figure(summary, args.figure)
    write_report(
        summary,
        classes,
        report=args.report,
        figure=args.figure,
        split_hash=split_hash,
    )
    for path in (summary_path, class_path, args.report, args.figure):
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
