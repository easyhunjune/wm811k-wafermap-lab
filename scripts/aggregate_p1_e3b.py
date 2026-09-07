from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_seeds import DEFAULT_SEEDS, FOCUS_LABELS, describe, load_runs, sha256

from wm811k.verdicts import class_tradeoff_verdict, count_verdict, macro_f1_verdict

EXPERIMENTS = ("E2", "E3", "E3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the P1 E3b comparison.")
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "repeat"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "P1_E3B.md"
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "p1_e3b_macro_f1_box.png",
    )
    return parser.parse_args()


def paired(
    frame: pd.DataFrame,
    *,
    value: str,
    left: str,
    right: str,
    index: list[str] | str = "seed",
) -> pd.Series:
    pivot = frame.pivot(index=index, columns="experiment", values=value)
    return pivot[left] - pivot[right]


def save_figure(summary: pd.DataFrame, path: Path) -> None:
    values = [
        summary.loc[summary["experiment"] == experiment, "macro_f1"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(values, tick_labels=EXPERIMENTS, showmeans=True)
    for index, run_values in enumerate(values, start=1):
        axis.scatter(np.full(len(run_values), index), run_values, alpha=0.7, zorder=3)
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Validation macro F1")
    axis.set_title("P1 E3b: softened class weighting")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def delta_text(values: pd.Series) -> str:
    return (
        f"{values.mean():+.4f} ± {values.std(ddof=1):.4f}; "
        f"양수 seed {int((values > 0).sum())}/{len(values)}"
    )


def count_delta_text(values: pd.Series) -> str:
    return (
        f"{values.mean():+.1f} ± {values.std(ddof=1):.1f}; "
        f"감소 seed {int((values < 0).sum())}/{len(values)}"
    )


def write_report(
    summary: pd.DataFrame,
    classes: pd.DataFrame,
    *,
    report: Path,
    figure: Path,
    split_hash: str,
) -> None:
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
        "# P1 E3b: 완화된 클래스 가중치",
        "",
        "## 사전 등록",
        "",
        "- 독립 변수: 클래스 가중치 공식",
        "- E2: 일반 CE",
        "- E3: `inverse_frequency`",
        "- E3b: `sqrt_inverse_frequency`",
        "- 고정 요소: 분할, 2채널 입력, 모델 구조와 나머지 학습 설정",
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
            f"![P1 E3b macro F1 box plot]({figure_link})",
            "",
            "## 대응 차이",
            "",
            f"- E3b−E2 macro F1: {delta_text(macro['E3b'] - macro['E2'])}",
            f"- E3b−E3 macro F1: {delta_text(macro['E3b'] - macro['E3'])}",
            (
                f"- E3b−E2 balanced accuracy: "
                f"{delta_text(balanced['E3b'] - balanced['E2'])}"
            ),
            (
                f"- E3b−E2 `none` false positive: "
                f"{count_delta_text(none_fp['E3b'] - none_fp['E2'])}"
            ),
            (
                f"- E3b−E3 `none` false positive: "
                f"{count_delta_text(none_fp['E3b'] - none_fp['E3'])}"
            ),
            "",
            "## 클래스별 정밀도·재현율 변화",
            "",
            "| 클래스 | E3b−E2 정밀도 | E3b−E2 재현율 | E3b−E3 정밀도 | E3b−E3 재현율 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in FOCUS_LABELS:
        p = precision.xs(label, level="label")
        r = recall.xs(label, level="label")
        lines.append(
            f"| {label} | {delta_text(p['E3b'] - p['E2'])} | "
            f"{delta_text(r['E3b'] - r['E2'])} | "
            f"{delta_text(p['E3b'] - p['E3'])} | "
            f"{delta_text(r['E3b'] - r['E3'])} |"
        )
    lines.extend(
        [
            "",
            "## 판정 규칙",
            "",
            "- E3b가 E3보다 정밀도 하락을 줄이면서 희소 클래스 재현율을 유지하는지 확인한다.",
            "- macro F1 또는 정밀도가 크게 하락하면 단일 클래스 재현율 상승만으로 개선 판정하지 않는다.",
            "- 평균 차이가 seed 변동보다 작거나 방향이 일관되지 않으면 우월성을 주장하지 않는다.",
            "",
            "## 결론",
            "",
            macro_f1_verdict("E3b−E3", macro["E3b"] - macro["E3"]),
            macro_f1_verdict("E3b−E2", macro["E3b"] - macro["E2"]),
            count_verdict("`none` false positive E3b−E2", none_fp["E3b"] - none_fp["E2"]),
            "",
            "### E3b−E3 클래스별 판정",
            "",
        ]
    )
    for label in FOCUS_LABELS:
        p = precision.xs(label, level="label")
        r = recall.xs(label, level="label")
        lines.append(
            class_tradeoff_verdict(
                label,
                recall_deltas=r["E3b"] - r["E3"],
                precision_deltas=p["E3b"] - p["E3"],
            )
        )
    lines.extend(
        [
            "",
            "### E3b−E2 클래스별 판정",
            "",
        ]
    )
    for label in FOCUS_LABELS:
        p = precision.xs(label, level="label")
        r = recall.xs(label, level="label")
        lines.append(
            class_tradeoff_verdict(
                label,
                recall_deltas=r["E3b"] - r["E2"],
                precision_deltas=p["E3b"] - p["E2"],
            )
        )
    lines.extend(
        [
            "",
            "## 다음 결정",
            "",
            (
                "E3b를 현재 P1의 선도 후보로 유지한다. 테스트 데이터는 갱신하지 않는다. "
                "Edge-Loc 정밀도와 희소 클래스 재현율의 추가 균형이 필요하면 다음 실험으로 "
                "가중치 상한을 사전 고정한 E3c를 비교한다."
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
    summary_path = args.input_dir / "p1_e3b_summary.csv"
    class_path = args.input_dir / "p1_e3b_class_metrics.csv"
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
