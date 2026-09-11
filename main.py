"""
영화 흥행 예측기
- kobis_movies.csv (영화별 표) 로 총 관객 수(total_audi)를 예측하는 다중 회귀 모델
- kobis_daily.csv (일별 박스오피스 표) 는 기준 기간 표시에만 사용
- movieCd(영화코드) 기준 정렬 후, 10편마다 앞의 3편을 테스트셋으로 분리
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

# 회귀에 쓸 수 있는 후보 수치형 변수
CANDIDATE_FEATURES = [
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "days_in_top10",
]
TARGET = "total_audi"
PRED_FLOOR = 1000  # 이보다 작은 예측값은 그래프 바닥에 붙여 표시

st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 박스오피스 데이터를 이용한 총 관객 수 다중 회귀 예측")


# ------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


try:
    daily_df, movies_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------
# 기준 기간 (일별 표에서 읽기)
# ------------------------------------------------------------------
date_col = "날짜"
if date_col in daily_df.columns:
    dates_numeric = pd.to_numeric(daily_df[date_col], errors="coerce").dropna().astype(int)
    start_raw = str(int(dates_numeric.min())).zfill(8)
    end_raw = str(int(dates_numeric.max())).zfill(8)

    def fmt_date(d):
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    period_text = f"{fmt_date(start_raw)} ~ {fmt_date(end_raw)}"
else:
    period_text = "확인 불가 (날짜 열을 찾을 수 없음)"

st.info(f"📅 기준 기간(일별 박스오피스 데이터 기준): **{period_text}**")

# ------------------------------------------------------------------
# 영화별 원본 표 상단 10행 그대로 표시
# ------------------------------------------------------------------
st.subheader("📋 영화별 원본 표 (상위 10행)")
st.dataframe(movies_df.head(10), use_container_width=True)

# ------------------------------------------------------------------
# movieCd 순 정렬 + 10편마다 앞 3편 테스트셋 분리
# ------------------------------------------------------------------
movies_sorted = movies_df.sort_values("movieCd").reset_index(drop=True)

# ------------------------------------------------------------------
# 변수 선택 (체크박스)
# ------------------------------------------------------------------
st.subheader("✅ 회귀에 사용할 변수 선택")

available_features = [c for c in CANDIDATE_FEATURES if c in movies_sorted.columns]

selected_features = []
checkbox_cols = st.columns(len(available_features)) if available_features else []
for col, feat in zip(checkbox_cols, available_features):
    with col:
        if st.checkbox(feat, value=True, key=f"chk_{feat}"):
            selected_features.append(feat)

if not selected_features:
    st.warning("최소 하나 이상의 변수를 선택해주세요.")
    st.stop()

# ------------------------------------------------------------------
# 학습 데이터 준비
# ------------------------------------------------------------------
use_cols = list(dict.fromkeys(selected_features + [TARGET]))
data = movies_sorted.copy()

for c in use_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=use_cols).reset_index(drop=True)

# 10편마다 앞의 3편을 테스트셋으로 분리 (movieCd 순 정렬된 순서 기준)
group_pos = data.index % 10
test_mask = group_pos < 3
train_mask = ~test_mask

train_df = data[train_mask].reset_index(drop=True)
test_df = data[test_mask].reset_index(drop=True)

if len(train_df) == 0 or len(test_df) == 0:
    st.error("학습 데이터 또는 평가 데이터가 부족합니다. 변수를 다시 선택해주세요.")
    st.stop()

X_train, y_train = train_df[selected_features], train_df[TARGET]
X_test, y_test = test_df[selected_features], test_df[TARGET]

# ------------------------------------------------------------------
# 모델 학습
# ------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

r2 = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

# ------------------------------------------------------------------
# 결과 요약
# ------------------------------------------------------------------
st.subheader("📊 모델 학습 결과")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("전체 영화 수", len(data))
c2.metric("학습에 쓴 영화 수", len(train_df))
c3.metric("평가에 쓴 영화 수", len(test_df))
c4.metric("R² (결정계수)", f"{r2:.3f}")
c5.metric("MAE (평균 절대 오차)", f"{mae:,.0f} 명")

st.caption(f"기준 기간: {period_text}  |  RMSE(평균 제곱근 오차): {rmse:,.0f} 명  |  사용 변수: {', '.join(selected_features)}")

with st.expander("회귀 계수 보기"):
    coef_df = pd.DataFrame({
        "변수": selected_features,
        "계수": model.coef_,
    })
    coef_df.loc[len(coef_df)] = ["절편(intercept)", model.intercept_]
    st.dataframe(coef_df, use_container_width=True)

# ------------------------------------------------------------------
# 산점도: 실제 vs 예측 (로그-로그, 대각선 포함)
# ------------------------------------------------------------------
st.subheader("📈 테스트 영화: 실제 총 관객 수 vs 예측 총 관객 수")

below_floor_mask = pred_test < PRED_FLOOR
n_below_floor = int(below_floor_mask.sum())

display_pred = np.clip(pred_test, PRED_FLOOR, None)
display_actual = np.clip(y_test.values, 1, None)  # 로그축을 위한 안전장치

name_col = "movieNm" if "movieNm" in test_df.columns else None
hover_names = test_df[name_col].values if name_col else [""] * len(test_df)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=display_actual,
        y=display_pred,
        mode="markers",
        name="테스트 영화",
        marker=dict(size=9, color="royalblue", opacity=0.75, line=dict(width=1, color="white")),
        customdata=np.stack([hover_names, y_test.values, pred_test], axis=-1),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "실제 총 관객: %{customdata[1]:,.0f}명<br>"
            "예측 총 관객: %{customdata[2]:,.0f}명<extra></extra>"
        ),
    )
)

axis_min = min(display_actual.min(), display_pred.min())
axis_max = max(display_actual.max(), display_pred.max())
diag = [axis_min, axis_max]

fig.add_trace(
    go.Scatter(
        x=diag,
        y=diag,
        mode="lines",
        name="실제 = 예측",
        line=dict(dash="dash", color="gray"),
    )
)

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그축)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그축)")
fig.update_layout(
    height=600,
    title="실제 총 관객 수 vs 예측 총 관객 수 (로그-로그)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

st.warning(
    f"예측값이 {PRED_FLOOR:,}명 미만으로 나온 영화는 그래프 바닥({PRED_FLOOR:,}명 지점)에 붙여 표시했습니다. "
    f"해당 영화는 총 **{n_below_floor}편**입니다."
)
