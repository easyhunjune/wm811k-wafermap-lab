from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.constants import LABELS
from wm811k.verdicts import class_tradeoff_verdict, count_verdict, macro_f1_verdict

EXPERIMENTS = ("E1", "E2", "E3")
DEFAULT_SEEDS = (42, 52, 62, 72, 82)
FOCUS_LABELS = ("Scratch", "Loc", "Edge-Loc", "Donut")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and aggregate P0 repeat-seed runs.")
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "repeat"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "SEED_REPEAT.md"
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "seed_macro_f1_box.png",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_runs(
    input_dir: Path,
    *,
    seeds: tuple[int, ...],
    expected_split_hash: str,
    experiments: tuple[str, ...] = EXPERIMENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing: list[str] = []
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    for experiment in experiments:
        for seed in seeds:
            run_dir = input_dir / experiment / f"seed{seed}"
            metrics_path = run_dir / "validation_metrics.json"
            meta_path = run_dir / "run_meta.json"
            absent = [str(path) for path in (metrics_path, meta_path) if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for source_name, payload in (("metrics", metrics), ("run_meta", meta)):
                if payload.get("experiment") != experiment or int(payload.get("seed", -1)) != seed:
                    raise ValueError(
                        f"{run_dir}: {source_name} experiment/seed metadata does not match its path."
                    )
                if payload.get("stage") != "validation":
                    raise ValueError(f"{run_dir}: {source_name} stage must be validation.")
                if payload.get("split_sha256") != expected_split_hash:
                    raise ValueError(
                        f"{run_dir}: {source_name} split hash differs from artifacts/splits.csv."
                    )
            if meta.get("status") != "completed":
                raise ValueError(f"{run_dir}: run status is {meta.get('status')!r}.")
            metric_values = metrics["metrics"]
            recalls = metric_values["per_class_recall"]
            if len(recalls) != len(LABELS):
                raise ValueError(f"{metrics_path}: expected {len(LABELS)} class recalls.")
            confusion = np.asarray(metric_values["confusion_matrix"], dtype=int)
            predicted_counts = confusion.sum(axis=0)
            precisions = np.divide(
                np.diag(confusion),
                predicted_counts,
                out=np.zeros(len(LABELS), dtype=float),
                where=predicted_counts != 0,
            )
            none_index = LABELS.index("none")
            none_false_positives = int(confusion[:, none_index].sum() - confusion[none_index, none_index])
            summary_rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "macro_f1": metric_values["macro_f1"],
                    "balanced_accuracy": metric_values["balanced_accuracy"],
                    "best_epoch": meta["best_epoch"],
                    "epochs_run": meta["epochs_run"],
                    "early_stopped": meta["early_stopped"],
                    "duration_seconds": meta["duration_seconds"],
                    "split_sha256": meta["split_sha256"],
                    "none_false_positives": none_false_positives,
                    "status": meta["status"],
                }
            )
            class_rows.extend(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                }
                for label, precision, recall in zip(
                    LABELS, precisions, recalls, strict=True
                )
            )
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing {len(missing)} required P0 files:\n{formatted}")
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows)


def describe(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def paired_delta(
    frame: pd.DataFrame,
    value: str,
    left: str,
    right: str,
) -> pd.Series:
    pivot = frame.pivot(index="seed", columns="experiment", values=value)
    return pivot[left] - pivot[right]


def format_stats(stats: dict[str, float]) -> str:
    return (
        f"{stats['mean']:.4f} ± {stats['std']:.4f} "
        f"({stats['min']:.4f}–{stats['max']:.4f}; median {stats['median']:.4f})"
    )


def write_report(
    summary: pd.DataFrame,
    class_metrics: pd.DataFrame,
    *,
    report_path: Path,
    figure_path: Path,
    split_hash: str,
    seeds: tuple[int, ...],
) -> None:
    try:
        figure_link = figure_path.relative_to(report_path.parent).as_posix()
    except ValueError:
        figure_link = figure_path.as_posix()
    experiment_stats = {
        experiment: describe(
            summary.loc[summary["experiment"] == experiment, "macro_f1"]
        )
        for experiment in EXPERIMENTS
    }
    e2_e1 = paired_delta(summary, "macro_f1", "E2", "E1")
    e3_e2 = paired_delta(summary, "macro_f1", "E3", "E2")
    recall_pivot = class_metrics.pivot(
        index=["seed", "label"], columns="experiment", values="recall"
    )
    precision_pivot = class_metrics.pivot(
        index=["seed", "label"], columns="experiment", values="precision"
    )
    fp_delta = paired_delta(summary, "none_false_positives", "E3", "E2")

    lines = [
        "# P0 반복 seed 재검증",
        "",
        "## 실험 질문과 사전 해석 규칙",
        "",
        "1. E2−E1의 macro F1 차이가 학습 seed 변동을 넘어 일관되는가?",
        "2. E3의 희소 클래스 재현율 상승과 정밀도 하락이 여러 seed에서 반복되는가?",
        "",
        (
            "n=5이므로 평균 차이가 seed 간 표준편차보다 작거나 방향이 일관되지 않으면 "
            "우월성을 주장하지 않는다. 이 보고서는 검증 데이터만 사용한다."
        ),
        "",
        "## 실행 조건",
        "",
        f"- 학습 seed: {', '.join(map(str, seeds))}",
        f"- 고정 분할 SHA-256: `{split_hash}`",
        (
            "- 고정 요소: 분할, 모델 구조, 손실, optimizer, 학습률, weight decay, "
            "batch size, 최대 epoch, 조기 종료 기준, 입력 크기"
        ),
        "",
        "## 검증 macro F1",
        "",
        "| 실험 | 평균 ± 표준편차 (최소–최대; 중앙값) |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {experiment} | {format_stats(experiment_stats[experiment])} |"
        for experiment in EXPERIMENTS
    )
    lines.extend(
        [
            "",
            f"![seed별 macro F1 box plot]({figure_link})",
            "",
            "## 대응 비교",
            "",
            (
                f"- E2−E1 macro F1: {format_stats(describe(e2_e1))}; "
                f"양수 seed {int((e2_e1 > 0).sum())}/{len(e2_e1)}"
            ),
            (
                f"- E3−E2 macro F1: {format_stats(describe(e3_e2))}; "
                f"양수 seed {int((e3_e2 > 0).sum())}/{len(e3_e2)}"
            ),
            "",
            "## E3−E2 클래스별 정밀도·재현율 변화",
            "",
            "| 클래스 | 정밀도 변화 | 정밀도 상승 | 재현율 변화 | 재현율 상승 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in FOCUS_LABELS:
        recall_delta = recall_pivot.xs(label, level="label")["E3"] - recall_pivot.xs(
            label, level="label"
        )["E2"]
        precision_delta = precision_pivot.xs(label, level="label")[
            "E3"
        ] - precision_pivot.xs(label, level="label")["E2"]
        lines.append(
            f"| {label} | {precision_delta.mean():+.4f} ± "
            f"{precision_delta.std(ddof=1):.4f} | "
            f"{int((precision_delta > 0).sum())}/{len(precision_delta)} | "
            f"{recall_delta.mean():+.4f} ± {recall_delta.std(ddof=1):.4f} | "
            f"{int((recall_delta > 0).sum())}/{len(recall_delta)} |"
        )
    lines.extend(
        [
            "",
            count_verdict("`none` false positive E3−E2", fp_delta),
            "",
            "## 실행 상태",
            "",
            f"- 완료: {int((summary['status'] == 'completed').sum())}/{len(summary)}",
            f"- 조기 종료: {int(summary['early_stopped'].sum())}/{len(summary)}",
            "- 실패 실행은 집계 전에 오류로 중단되므로 이 보고서에 누락된 채 반영될 수 없다.",
            "",
            "## 판정",
            "",
            macro_f1_verdict("E2−E1", e2_e1),
            macro_f1_verdict("E3−E2", e3_e2),
            "",
            "### E3−E2 클래스별 판정",
            "",
        ]
    )
    for label in FOCUS_LABELS:
        recall_label = recall_pivot.xs(label, level="label")
        precision_label = precision_pivot.xs(label, level="label")
        lines.append(
            class_tradeoff_verdict(
                label,
                recall_deltas=recall_label["E3"] - recall_label["E2"],
                precision_deltas=precision_label["E3"] - precision_label["E2"],
            )
        )
    lines.extend(
        [
            "",
            "### 주장할 수 없는 것",
            "",
            "- E2 입력 표현의 일반적인 우월성",
            "- E3가 전체적으로 더 좋은 모델이라는 주장",
            "- 검증 반복 결과에 근거한 테스트 또는 운영 환경 성능 개선",
            "",
            "## 다음 결정",
            "",
            (
                "P1에 진입한다. 첫 비교는 E3b `sqrt(inverse_frequency)` 가중치이며, "
                "E2·E3와 같은 분할·5개 seed를 사용해 정밀도 하락을 줄이면서 "
                "희소 클래스 재현율을 유지하는지 검증한다. 테스트 데이터는 사용하지 않는다."
            ),
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_boxplot(summary: pd.DataFrame, figure_path: Path) -> None:
    values = [
        summary.loc[summary["experiment"] == experiment, "macro_f1"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(values, tick_labels=EXPERIMENTS, showmeans=True)
    for index, run_values in enumerate(values, start=1):
        axis.scatter(np.full(len(run_values), index), run_values, alpha=0.7, zorder=3)
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Validation macro F1")
    axis.set_title("P0 repeat-seed validation")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates.")
    split_hash = sha256(args.splits)
    summary, class_metrics = load_runs(
        args.input_dir,
        seeds=seeds,
        expected_split_hash=split_hash,
    )
    summary_path = args.input_dir / "seed_summary.csv"
    class_path = args.input_dir / "seed_class_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    class_metrics.to_csv(class_path, index=False)
    save_boxplot(summary, args.figure)
    write_report(
        summary,
        class_metrics,
        report_path=args.report,
        figure_path=args.figure,
        split_hash=split_hash,
        seeds=seeds,
    )
    for path in (summary_path, class_path, args.report, args.figure):
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
