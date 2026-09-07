"""Compare one experiment trained under both class-weight normalizations.

`mean_one` sets the arithmetic mean of the nine class weights to 1; `sample_mean_one`
sets the mean over training samples to 1. The two differ only by a constant factor, so
the weight *shape* is identical and the comparison isolates the effective loss scale
that reaches the optimizer — and with it the effective learning rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_seeds import DEFAULT_SEEDS, FOCUS_LABELS

from wm811k.constants import LABELS
from wm811k.verdicts import (
    class_tradeoff_verdict,
    count_verdict,
    metric_verdict,
    spread_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="E3b")
    parser.add_argument(
        "--baseline-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "repeat"
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "norm_probe" / "repeat",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to reports/WEIGHT_NORMALIZATION_PROBE_{EXPERIMENT}.md.",
    )
    args = parser.parse_args()
    if args.report is None:
        name = f"WEIGHT_NORMALIZATION_PROBE_{args.experiment.upper()}.md"
        args.report = PROJECT_ROOT / "reports" / name
    return args


def load_run(run_dir: Path) -> dict:
    path = run_dir / "validation_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing probe or baseline run: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_commit(run_dir: Path) -> str:
    """Report the code version a run was produced with, as run_meta.json recorded it."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return "미기록"
    commit = json.loads(meta_path.read_text(encoding="utf-8")).get("git_commit")
    return f"`{commit[:7]}`" if commit else "미기록(`git_commit: null`)"


def precision(confusion: list[list[float]], index: int) -> float:
    matrix = np.asarray(confusion, dtype=float)
    predicted = matrix[:, index].sum()
    return float(matrix[index, index] / predicted) if predicted else 0.0


def none_false_positives(confusion: list[list[float]]) -> float:
    matrix = np.asarray(confusion, dtype=float)
    index = LABELS.index("none")
    return float(matrix[:, index].sum() - matrix[index, index])


def effective_scale(run: dict) -> float:
    weight_metadata = run["class_weight"]
    if "effective_sample_scale" in weight_metadata:
        return float(weight_metadata["effective_sample_scale"])
    counts = list(weight_metadata["counts"].values())
    return float(np.average(list(weight_metadata["weights"].values()), weights=counts))


def build_report(experiment: str, seeds: list[int], baseline_dir: Path, probe_dir: Path) -> str:
    macro_deltas: list[float] = []
    balanced_deltas: list[float] = []
    false_positive_deltas: list[float] = []
    per_label: dict[str, dict[str, list[float]]] = {
        label: {"recall": [], "precision": []} for label in FOCUS_LABELS
    }
    rows: list[str] = []
    scales: dict[str, float] = {}
    baseline_commit = run_commit(baseline_dir / experiment / f"seed{seeds[0]}")
    probe_commit = run_commit(probe_dir / experiment / f"seed{seeds[0]}")
    baseline_macro: list[float] = []
    probe_macro: list[float] = []

    for seed in seeds:
        base = load_run(baseline_dir / experiment / f"seed{seed}")
        probe = load_run(probe_dir / experiment / f"seed{seed}")
        scales = {
            base["class_weight"].get("normalization", "mean_one"): effective_scale(base),
            probe["class_weight"]["normalization"]: effective_scale(probe),
        }
        base_metrics, probe_metrics = base["metrics"], probe["metrics"]
        baseline_macro.append(base_metrics["macro_f1"])
        probe_macro.append(probe_metrics["macro_f1"])
        macro_deltas.append(probe_metrics["macro_f1"] - base_metrics["macro_f1"])
        balanced_deltas.append(
            probe_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
        )
        false_positive_deltas.append(
            none_false_positives(probe_metrics["confusion_matrix"])
            - none_false_positives(base_metrics["confusion_matrix"])
        )
        for label in FOCUS_LABELS:
            index = LABELS.index(label)
            per_label[label]["recall"].append(
                probe_metrics["per_class_recall"][index]
                - base_metrics["per_class_recall"][index]
            )
            per_label[label]["precision"].append(
                precision(probe_metrics["confusion_matrix"], index)
                - precision(base_metrics["confusion_matrix"], index)
            )
        rows.append(
            f"| {seed} | {base_metrics['macro_f1']:.4f} | {probe_metrics['macro_f1']:.4f} | "
            f"{macro_deltas[-1]:+.4f} | {base['best_epoch']} | {probe['best_epoch']} |"
        )

    lines = [
        f"# 클래스 가중치 정규화 프로브 ({experiment})",
        "",
        "## 질문",
        "",
        (
            "`mean_one`은 9개 클래스 가중치의 산술평균을 1로 맞춘다. 배치 평균 손실을 쓰므로 "
            "옵티마이저가 보는 실효 스케일은 표본 기준 가중 평균이 되어 1보다 훨씬 작아진다. "
            "`sample_mean_one`은 이 표본 기준 평균을 1로 맞춰 비가중 실험과 손실 스케일을 "
            "맞춘다. 두 방식은 상수배 관계라 가중치 **모양**은 같고, 따라서 이 비교는 "
            "실효 학습률 차이만을 분리한다."
        ),
        "",
        "## 실행 조건",
        "",
        f"- 실험: {experiment}, 학습 seed: {', '.join(map(str, seeds))}",
        "- 분할·모델·손실 공식·optimizer·학습률·나머지 학습 설정 고정",
        "- 검증 데이터만 사용. 테스트 데이터는 사용하지 않았다.",
        (
            f"- **코드 버전은 두 arm이 다르다**: baseline {baseline_commit}, "
            f"프로브 {probe_commit}. 프로브 arm은 체크포인트 저장 조건에서 `min_delta`를 "
            "분리한 뒤의 코드이며, 이 변경은 더 높은 macro F1을 저장하므로 편향 방향이 "
            "프로브 arm에 유리하다."
        ),
        "",
        "| 정규화 | 실효 손실 스케일 |",
        "|---|---:|",
    ]
    for name, value in scales.items():
        lines.append(f"| `{name}` | {value:.4f} |")
    lines.extend(
        [
            "",
            "## seed별 검증 macro F1",
            "",
            "| seed | `mean_one` | `sample_mean_one` | 차이 | best epoch (전) | best epoch (후) |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 판정",
            "",
            metric_verdict("`sample_mean_one` − `mean_one`", "macro F1", macro_deltas),
            metric_verdict(
                "`sample_mean_one` − `mean_one`", "balanced accuracy", balanced_deltas
            ),
            count_verdict("`none` false positive", false_positive_deltas),
            spread_line(
                "macro F1",
                baseline_macro,
                probe_macro,
                before_name="mean_one",
                after_name="sample_mean_one",
            ),
            "",
            "### 클래스별 판정",
            "",
        ]
    )
    for label in FOCUS_LABELS:
        lines.append(
            class_tradeoff_verdict(
                label,
                recall_deltas=per_label[label]["recall"],
                precision_deltas=per_label[label]["precision"],
            )
        )
    lines.extend(
        [
            "",
            "## 이 프로브가 말하지 않는 것",
            "",
            (
                f"- {experiment} 외 실험. 실효 스케일 축소 폭은 가중치 공식마다 다르므로"
                + (
                    " 여기서 얻은 결과를 다른 실험에 그대로 적용할 수 없다."
                    if experiment == "E3"
                    else " 축소가 더 큰 E3에 여기서 얻은 결과를 그대로 적용할 수 없다."
                )
            ),
            "- 테스트 성능. 검증 데이터만 사용했다.",
            "- 학습률을 함께 조정했을 때의 결과. 학습률은 고정하고 손실 스케일만 바꿨다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = build_report(args.experiment, args.seeds, args.baseline_dir, args.probe_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"saved: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
