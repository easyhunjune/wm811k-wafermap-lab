from pathlib import Path

import pandas as pd
import pytest

from scripts.build_test_report import describe_run
from wm811k.verdicts import (
    class_tradeoff_verdict,
    count_verdict,
    leading_candidate_line,
    macro_f1_verdict,
    metric_verdict,
    spread_line,
    summarise_delta,
)


def test_summarise_delta_counts_both_directions():
    verdict = summarise_delta([0.1, -0.2, 0.3, 0.0])

    assert verdict.total == 4
    assert verdict.positive == 2
    assert verdict.negative == 1
    assert not verdict.unanimous


def test_summarise_delta_needs_more_than_one_seed():
    with pytest.raises(ValueError, match="at least two seeds"):
        summarise_delta([0.1])


def test_a_unanimous_gain_beyond_the_spread_is_called_an_improvement():
    line = macro_f1_verdict("E3b−E3", [0.03, 0.04, 0.035, 0.045, 0.03])

    assert "상승 5/5 seed" in line
    assert "개선으로 판정한다" in line


def test_a_unanimous_loss_beyond_the_spread_is_not_called_an_improvement():
    line = macro_f1_verdict("E3−E2", [-0.03, -0.04, -0.02, -0.05, -0.01])

    assert "상승 0/5 seed" in line
    assert "하락으로 판정한다" in line
    assert "개선으로 판정한다" not in line


def test_a_split_direction_is_not_called_consistent():
    line = macro_f1_verdict("E2−E1", [0.001, -0.002, 0.003, 0.001, 0.002])

    assert "일관된 차이로 판정하지 않는다" in line


def test_denominators_follow_the_number_of_seeds():
    three = macro_f1_verdict("E3b−E3", [0.03, 0.04, 0.05])
    seven = macro_f1_verdict("E3b−E3", [0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04])

    assert "상승 3/3 seed" in three
    assert "상승 7/7 seed" in seven


def test_class_tradeoff_reports_recall_gain_against_precision_loss():
    line = class_tradeoff_verdict(
        "Scratch",
        recall_deltas=[0.10, 0.09, 0.11, 0.08, 0.12],
        precision_deltas=[-0.15, -0.16, -0.14, -0.15, -0.17],
    )

    assert "재현율 상승과 정밀도 하락이 모든 seed에서 반복됐다" in line
    assert "상승 5/5 seed" in line
    assert "하락 5/5 seed" in line


def test_class_tradeoff_does_not_call_a_mean_decline_a_recall_gain():
    # Three seeds nudge recall up, but one large drop pulls the average down.
    line = class_tradeoff_verdict(
        "Loc",
        recall_deltas=[0.02, 0.03, 0.01, -0.30, -0.16],
        precision_deltas=[-0.02, -0.03, -0.01, 0.20, 0.11],
    )

    assert "재현율 -0.0800 (하락 2/5 seed)" in line
    assert "재현율 상승과 정밀도 하락의 절충 패턴은 확인되지 않는다" in line


def test_the_tradeoff_pattern_needs_the_means_to_agree_not_just_the_counts():
    # Recall falls on average and precision falls on average, so there is no
    # recall-for-precision trade. Counting seeds alone would find a majority of each and
    # print a trade-off, contradicting the means on the same line.
    line = class_tradeoff_verdict(
        "Loc",
        recall_deltas=[0.01, -0.02, -0.03, -0.04, 0.02],
        precision_deltas=[-0.05, -0.06, 0.01, 0.02, -0.07],
    )

    assert "재현율 -0.0120 (하락 3/5 seed)" in line
    assert "절충 패턴은 확인되지 않는다" in line
    assert "과반" not in line


def test_the_tradeoff_pattern_counts_seeds_where_both_moved_together():
    # Recall rises in 3 of 5 and precision falls in 3 of 5, but they coincide in only 2:
    # separate majorities are not a majority of the trade-off.
    line = class_tradeoff_verdict(
        "Donut",
        recall_deltas=[0.05, 0.04, 0.03, -0.02, -0.02],
        precision_deltas=[-0.05, -0.04, 0.03, 0.02, -0.02],
    )

    assert "절충 패턴은 확인되지 않는다" in line


def test_a_metric_that_moved_the_same_way_in_every_seed_is_still_reported():
    # The trade-off pattern is absent, but precision rose in all five seeds. Saying only
    # "the direction varies" would contradict the numbers printed on the same line.
    line = class_tradeoff_verdict(
        "Loc",
        recall_deltas=[-0.09, -0.08, 0.02, -0.11, -0.10],
        precision_deltas=[0.21, 0.23, 0.19, 0.25, 0.25],
    )

    assert "정밀도 +0.2260 (상승 5/5 seed)" in line
    assert "다만 정밀도는 모든 seed에서 같은 방향이다" in line


def test_a_metric_that_never_moved_is_not_called_inconsistent():
    line = metric_verdict("A−B", "macro F1", [0.0, 0.0, 0.0])

    assert "모든 seed에서 차이가 0이다" in line
    assert "갈린다" not in line

    counts = count_verdict("`none` false positive", [0.0, 0.0, 0.0])

    assert "모든 seed에서 차이가 0이다" in counts
    assert "갈린다" not in counts


def test_spread_line_does_not_name_a_direction_for_a_negligible_change():
    line = spread_line(
        "macro F1",
        [0.850, 0.860, 0.855],
        [0.851, 0.861, 0.856],
        before_name="a",
        after_name="b",
    )

    assert "사실상 같다" in line


def test_leading_candidate_needs_more_than_one_seed_per_candidate():
    macro = pd.DataFrame({"E3b": [0.86], "E3c": [0.85]})

    with pytest.raises(ValueError, match="at least two seeds"):
        leading_candidate_line(macro, ("E3b", "E3c"))


def test_leading_candidate_reports_a_tie_instead_of_inventing_an_order():
    macro = pd.DataFrame({"E3b": [0.85, 0.86], "E3c": [0.86, 0.85]})

    line = leading_candidate_line(macro, ("E3b", "E3c"))

    assert "같다" in line
    assert "가릴 수 없다" in line


def test_leading_candidate_needs_two_candidates():
    macro = pd.DataFrame({"E3b": [0.85, 0.86]})

    with pytest.raises(ValueError, match="at least two candidates"):
        leading_candidate_line(macro, ("E3b",))


def test_leading_candidate_claims_only_the_metric_it_computed():
    macro = pd.DataFrame({"E3b": [0.86, 0.85, 0.86], "E3c": [0.85, 0.84, 0.86]})

    line = leading_candidate_line(macro, ("E3b", "E3c"))

    assert "정밀도" not in line
    assert "재현율" not in line


def test_spread_line_survives_a_zero_standard_deviation():
    line = spread_line(
        "macro F1",
        [0.83, 0.82, 0.84],
        [0.80, 0.80, 0.80],
        before_name="a",
        after_name="b",
    )

    assert "비교 불가" in line


def test_spread_line_reports_how_run_to_run_variation_changed():
    line = spread_line(
        "macro F1",
        [0.83, 0.82, 0.83, 0.82, 0.80],
        [0.72, 0.81, 0.84, 0.79, 0.78],
        before_name="mean_one",
        after_name="sample_mean_one",
    )

    assert "seed 간 표준편차" in line
    assert "커졌다" in line


def test_class_tradeoff_flags_a_direction_that_varies_by_seed():
    line = class_tradeoff_verdict(
        "Donut",
        recall_deltas=[0.05, -0.02, 0.04, -0.01, 0.03],
        precision_deltas=[0.01, -0.02, 0.03, -0.04, 0.02],
    )

    assert "절충 패턴은 확인되지 않는다" in line
    assert "모든 seed에서 같은 방향이다" not in line


def test_count_verdict_reports_the_direction_and_its_consistency():
    line = count_verdict("`none` false positive E3b−E2", [-180.0, -190.0, -170.0])

    assert "감소" in line
    assert "감소 3/3 seed" in line
    assert "± 10.0" in line
    assert "모든 seed에서 같은 방향이다" in line


def test_count_verdict_counts_the_seeds_that_agree_with_the_mean():
    line = count_verdict("`none` false positive", [60.0, 80.0, -20.0])

    assert "증가 2/3 seed" in line
    assert "감소" not in line
    assert "방향이 seed마다 갈린다" in line


def test_metric_verdict_names_the_metric_it_was_given():
    line = metric_verdict("A−B", "balanced accuracy", [0.01, -0.02, 0.03])

    assert "A−B balanced accuracy:" in line
    assert "macro F1" not in line


def test_leading_candidate_is_the_higher_mean():
    macro = pd.DataFrame({"E3b": [0.86, 0.85, 0.86], "E3c": [0.85, 0.84, 0.86]})

    line = leading_candidate_line(macro, ("E3b", "E3c"))

    assert line.startswith("- E3b는 E3c보다")
    assert "E3b를 선도 후보로 둔다" in line


def test_leading_candidate_does_not_depend_on_argument_order():
    macro = pd.DataFrame({"E3b": [0.86, 0.85, 0.86], "E3c": [0.85, 0.84, 0.86]})

    assert leading_candidate_line(macro, ("E3c", "E3b")) == leading_candidate_line(
        macro, ("E3b", "E3c")
    )


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        ("artifacts/E2/test_predictions.csv", ("E2 Test", "e2_test")),
        ("artifacts/E3/validation_predictions.csv", ("E3 Validation", "e3_validation")),
        (
            "artifacts/repeat/E3b/seed52/validation_predictions.csv",
            ("E3b seed52 Validation", "e3b_seed52_validation"),
        ),
    ],
)
def test_report_titles_follow_the_predictions_file(predictions, expected):
    assert describe_run(Path(predictions)) == expected
