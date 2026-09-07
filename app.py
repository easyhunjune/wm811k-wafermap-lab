from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from wm811k.constants import LABELS
from wm811k.dashboard_data import (
    load_dashboard_tables,
    required_dashboard_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(
    page_title="WM-811K WaferMap Lab",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, object]:
    return load_dashboard_tables(PROJECT_ROOT)


missing = [path for path in required_dashboard_paths(PROJECT_ROOT) if not path.is_file()]
if missing:
    st.error("대시보드에 필요한 결과 파일이 없습니다.")
    st.code("\n".join(str(path) for path in missing))
    st.stop()

data = load_data()
validation = data["validation"]
repeat_validation = data["repeat_validation"]
test = data["test"]
class_report = data["class_report"]
radial = data["radial"]
edge = data["edge"]
cases = data["cases"]
split = data["split"]
audit = data["audit"]

st.title("WM-811K WaferMap Lab")
st.caption("누수 방지 분할, 불균형 평가, 오류 사례와 공간 분포를 한 화면에서 확인합니다.")
st.info(
    "테스트 결과를 확인한 뒤 모델을 다시 선택하지 않았습니다. "
    "`none`은 물리적 무결함이 아니라 라벨이 있는 비패턴 클래스입니다.",
    icon="ℹ️",
)

test_ci = test["cluster_bootstrap_ci"]["macro_f1"]
metric_columns = st.columns(3)
metric_columns[0].metric("Test macro F1", f"{test['metrics']['macro_f1']:.4f}")
metric_columns[0].caption(
    "로트 부트스트랩 95% CI: "
    f"{test_ci['lower']:.4f}–{test_ci['upper']:.4f}"
)
metric_columns[1].metric(
    "Balanced accuracy",
    f"{test['metrics']['balanced_accuracy']:.4f}",
)
metric_columns[2].metric("Test samples", f"{split['audit']['rows']['test']:,}")

tabs = st.tabs(
    [
        "실험 비교",
        "테스트 성능",
        "오류 사례",
        "공간 분석",
        "방법론",
    ]
)

with tabs[0]:
    st.subheader("단일 시드 E0~E3 검증 비교")
    melted = validation.melt(
        id_vars=["experiment"],
        value_vars=["macro_f1", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("experiment:N", title="Experiment"),
            xOffset="metric:N",
            y=alt.Y("score:Q", title="Validation score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N",
                title="Metric",
                scale=alt.Scale(
                    domain=["macro_f1", "balanced_accuracy"],
                    range=["#2563eb", "#f59e0b"],
                ),
            ),
            tooltip=[
                "experiment:N",
                "metric:N",
                alt.Tooltip("score:Q", format=".4f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")
    display_validation = validation.copy()
    display_validation["best_epoch"] = display_validation["best_epoch"].map(
        lambda value: "-" if pd.isna(value) else str(int(value))
    )
    display_validation["selected"] = display_validation["selected"].map(
        {True: "최종 선택", False: ""}
    )
    st.dataframe(
        display_validation.rename(
            columns={
                "experiment": "실험",
                "best_epoch": "최적 epoch",
                "macro_f1": "Macro F1",
                "balanced_accuracy": "Balanced accuracy",
                "selected": "선택",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
            "Balanced accuracy": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.markdown(
        "E2의 검증 macro F1이 가장 높아 테스트 전에 최종 후보로 고정했습니다. "
        "E1과의 차이는 0.0018로 작으므로 2채널 입력의 우월성을 단정하지 않습니다."
    )
    if not repeat_validation.empty:
        st.subheader("5개 시드 반복 검증")
        st.dataframe(
            repeat_validation.rename(
                columns={
                    "experiment": "실험",
                    "seeds": "시드 수",
                    "macro_f1_mean": "Macro F1 평균",
                    "macro_f1_std": "Macro F1 표준편차",
                    "balanced_accuracy_mean": "Balanced accuracy 평균",
                    "balanced_accuracy_std": "Balanced accuracy 표준편차",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Macro F1 평균": st.column_config.NumberColumn(format="%.4f"),
                "Macro F1 표준편차": st.column_config.NumberColumn(format="%.4f"),
                "Balanced accuracy 평균": st.column_config.NumberColumn(
                    format="%.4f"
                ),
                "Balanced accuracy 표준편차": st.column_config.NumberColumn(
                    format="%.4f"
                ),
            },
        )
        st.caption(
            "E3b·E3c는 후속 검증 결과이며 테스트 세트를 열기 전의 "
            "E2 선택을 소급해 바꾸지 않습니다."
        )

with tabs[1]:
    st.subheader("E2 최종 테스트")
    report_view = class_report[["class", "precision", "recall", "f1-score", "support"]].copy()
    recall_chart = (
        alt.Chart(report_view)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("class:N", title=None, sort="-x"),
            tooltip=[
                "class:N",
                alt.Tooltip("precision:Q", format=".4f"),
                alt.Tooltip("recall:Q", format=".4f"),
                alt.Tooltip("f1-score:Q", format=".4f"),
                alt.Tooltip("support:Q", format=",.0f"),
            ],
        )
        .properties(height=380)
    )
    left, right = st.columns([1, 1.15])
    with left:
        st.altair_chart(recall_chart, width="stretch")
        st.dataframe(
            report_view.rename(
                columns={
                    "class": "클래스",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1-score": "F1",
                    "support": "표본 수",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
                "F1": st.column_config.NumberColumn(format="%.4f"),
                "표본 수": st.column_config.NumberColumn(format="%d"),
            },
        )
    with right:
        st.image(
            str(FIGURES / "e2_test_confusion_matrix.png"),
            caption="행 정규화 혼동 행렬",
            width="stretch",
        )
    st.warning(
        "`Scratch`의 41%, `Loc`의 31%, `Edge-Loc`의 30%가 `none`으로 분류됐습니다. "
        "`Near-full` recall 1.0은 test 표본 32개에 대한 결과이므로 과신하지 않습니다.",
        icon="⚠️",
    )
    near_full = test["cluster_bootstrap_ci"]["per_class_recall"]["Near-full"]
    iid_reference = near_full.get("iid_exact_binomial_reference")
    if iid_reference:
        st.caption(
            "Near-full의 로트 군집 부트스트랩 구간은 관측 오답이 없어 "
            "1.000–1.000입니다. 웨이퍼 독립을 가정한 보조 "
            "Clopper–Pearson 95% 구간은 "
            f"{iid_reference['lower']:.3f}–{iid_reference['upper']:.3f}입니다."
        )

with tabs[2]:
    st.subheader("클래스별 정답·오답 사례")
    if cases is None:
        st.info(
            "사례 매니페스트와 원본 웨이퍼 이미지는 개인정보·용량 정책상 "
            "저장소에 포함되지 않습니다. 나머지 성능·공간 분석 탭은 정상 동작합니다."
        )
    else:
        selected_class = st.selectbox("실제 클래스", LABELS, index=7)
        case_type = st.radio(
            "사례 유형",
            options=["error", "correct"],
            format_func=lambda value: "오답" if value == "error" else "정답",
            horizontal=True,
        )
        slug = selected_class.lower().replace("-", "_")
        image_path = FIGURES / f"e2_{slug}_{case_type}_cases.png"
        if image_path.is_file():
            st.image(str(image_path), width="stretch")
        else:
            st.info("선택한 사례 그림은 이 로컬 환경에 없습니다.")
        selected_cases = cases[
            (cases["true_label"] == selected_class)
            & (cases["case_type"] == case_type)
        ][["source_row_id", "lotName", "true_label", "predicted_label"]]
        st.dataframe(
            selected_cases.rename(
                columns={
                    "source_row_id": "원본 행",
                    "lotName": "로트",
                    "true_label": "실제 라벨",
                    "predicted_label": "예측 라벨",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "고정 시드 42로 클래스별 최대 10개를 무작위 추출했습니다. "
            "Near-full은 관찰된 오답이 없습니다."
        )

with tabs[3]:
    st.subheader("정규화 반경 기반 공간 분석")
    spatial_class = st.selectbox("공간 프로파일 클래스", LABELS, index=3)
    selected_radial = radial[radial["class"] == spatial_class].copy()
    selected_radial["radius"] = (
        selected_radial["radius_start"] + selected_radial["radius_end"]
    ) / 2
    radial_chart = (
        alt.Chart(selected_radial)
        .mark_line(point=True, color="#2563eb")
        .encode(
            x=alt.X("radius:Q", title="Normalized radius r/R", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "defect_density:Q",
                title="Die-weighted defect density",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("radius_start:Q", format=".1f"),
                alt.Tooltip("radius_end:Q", format=".1f"),
                alt.Tooltip("defect_density:Q", format=".4f"),
                alt.Tooltip("die_count:Q", format=","),
            ],
        )
        .properties(height=360)
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.altair_chart(radial_chart, width="stretch")
    with right:
        selected_edge = edge[edge["class"] == spatial_class][
            ["zone", "radius_threshold", "die_count", "defect_count", "defect_density"]
        ]
        st.dataframe(
            selected_edge.rename(
                columns={
                    "zone": "영역",
                    "radius_threshold": "r/R 기준",
                    "die_count": "다이 수",
                    "defect_count": "불량 다이 수",
                    "defect_density": "결함 밀도",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "r/R 기준": st.column_config.NumberColumn(format="%.2f"),
                "다이 수": st.column_config.NumberColumn(format="%d"),
                "불량 다이 수": st.column_config.NumberColumn(format="%d"),
                "결함 밀도": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption("실제 mm 환산이 아니라 가장 먼 다이를 1로 둔 정규화 반경입니다.")
    with st.expander("전체 클래스 프로파일 보기"):
        st.image(str(FIGURES / "e2_test_radial_profiles.png"), width="stretch")
        st.image(str(FIGURES / "e2_test_edge_zones.png"), width="stretch")
    st.info(
        "notch 또는 결정 방향 정보가 없어 각도 분석은 제외했습니다. "
        "웨이퍼 맵만으로 공정·장비의 근본 원인을 확정하지 않습니다.",
        icon="ℹ️",
    )

with tabs[4]:
    st.subheader("누수 방지 평가 설계")
    flow_columns = st.columns(4)
    flow_columns[0].markdown("**1. 데이터 감사**  \n빈 라벨과 `none` 분리")
    flow_columns[1].markdown("**2. 로트 분할**  \n60/20/20, 중복 0")
    flow_columns[2].markdown("**3. 검증 선택**  \nMacro F1 기준")
    flow_columns[3].markdown("**4. 테스트**  \n고정 후 1회")

    split_rows = pd.DataFrame(
        {
            "분할": ["Train", "Validation", "Test"],
            "표본 수": [
                split["audit"]["rows"]["train"],
                split["audit"]["rows"]["validation"],
                split["audit"]["rows"]["test"],
            ],
            "로트 수": [
                split["audit"]["lots"]["train"],
                split["audit"]["lots"]["validation"],
                split["audit"]["lots"]["test"],
            ],
            "누락 클래스": [
                len(split["audit"]["missing_classes"]["train"]),
                len(split["audit"]["missing_classes"]["validation"]),
                len(split["audit"]["missing_classes"]["test"]),
            ],
        }
    )
    st.dataframe(split_rows, hide_index=True, width="stretch")
    st.markdown(
        f"""
        - 전체 행: **{audit["rows"]:,}개**
        - 라벨 데이터: **{audit["label_audit"]["recognized_labeled_rows"]:,}개**
        - 미라벨 데이터: **{audit["label_audit"]["unlabeled_rows"]:,}개**
        - 웨이퍼 맵 크기: **{audit["wafer_map_audit"]["distinct_shapes"]}종**
        - 관찰 값: **{audit["wafer_map_audit"]["observed_values"]}**
        """
    )
    st.markdown(
        """
        **해석 제한**

        - 최종 테스트 모델은 테스트를 열기 전에 고정했으며, 후속 5시드
          검증은 선택을 소급 변경하는 데 사용하지 않습니다.
        - 로트 단위 부트스트랩으로 동일 로트 내 상관을 반영했습니다.
        - `none`은 비패턴 클래스이지 물리적 정상 판정이 아닙니다.
        - 물리 메타데이터가 없어 edge 영역을 mm로 표현하지 않습니다.
        """
    )
