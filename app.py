import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import date
import glob
import os

st.set_page_config(page_title="Cannabis Market Estimates", layout="wide", page_icon="📈")

# ── Constants ─────────────────────────────────────────────────────────────────

LEGALIZATION_DATES = {
    "Alaska": "2014-11-04", "Arizona": "2020-11-03", "Arkansas": "2016-11-08",
    "California": "2016-11-08", "Colorado": "2012-11-06", "Connecticut": "2021-06-22",
    "Delaware": "2023-04-23", "District of Columbia": "2020-11-04",
    "Florida": "2016-11-08", "Hawaii": "2000-06-14", "Illinois": "2019-06-25",
    "Iowa": "2014-05-30", "Louisiana": "2015-06-29", "Maine": "2016-11-08",
    "Maryland": "2022-11-08", "Massachusetts": "2016-11-08", "Michigan": "2018-11-06",
    "Minnesota": "2023-05-30", "Mississippi": "2022-02-02", "Missouri": "2022-11-08",
    "Montana": "2020-11-03", "Nevada": "2016-11-08", "New Hampshire": "2013-07-23",
    "New Jersey": "2020-11-03", "New Mexico": "2021-04-12", "New York": "2021-03-31",
    "North Carolina": "2014-07-03", "North Dakota": "2016-11-08", "Ohio": "2023-11-07",
    "Oklahoma": "2018-06-26", "Oregon": "2014-11-04", "Pennsylvania": "2016-04-17",
    "Rhode Island": "2022-05-25", "South Dakota": "2020-11-03", "Utah": "2018-11-06",
    "Vermont": "2018-01-22", "Virginia": "2024-04-07", "Washington": "2012-11-06",
    "West Virginia": "2017-04-19",
}

REC_STATES = {
    "Alaska", "Arizona", "California", "Colorado", "Connecticut", "Illinois",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Missouri", "Montana",
    "Nevada", "New Jersey", "New Mexico", "New York", "Oregon", "Rhode Island",
    "Vermont", "Washington",
}

STATES_USE_GROWTH_MODEL = {
    "Oregon", "Massachusetts", "Maryland", "Maine", "California", "North Dakota",
    "Utah", "Hawaii", "New Mexico", "Minnesota", "Montana", "Rhode Island",
    "Missouri", "Pennsylvania", "Michigan", "Colorado", "District of Columbia",
    "Virginia", "New York", "Connecticut", "Arizona", "Mississippi", "Ohio",
    "New Jersey", "Nevada", "Vermont", "Alaska", "South Dakota", "Arkansas",
    "West Virginia", "Illinois", "Oklahoma", "Washington", "Florida",
    "Delaware", "New Hampshire",
}

GROWTH_TRAINING_STATES = [
    "Oregon", "Massachusetts", "Maryland", "Maine", "California", "North Dakota",
    "Utah", "Hawaii", "New Mexico", "Minnesota", "Montana", "Rhode Island",
    "Missouri", "Pennsylvania", "Michigan", "Colorado", "District of Columbia",
    "Virginia", "New York", "Connecticut", "Arizona", "Mississippi", "Ohio",
    "New Jersey", "Nevada", "Vermont", "Alaska", "South Dakota", "Arkansas",
    "West Virginia",
]

# States with token legal status but no functioning retail market —
# show Year 1/2/... projections rather than calendar years
NO_MARKET_STATES = {"Texas", "Virginia"}

ALL_MODEL_STATES = sorted([
    "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Hawaii", "Illinois",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Dakota", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Nebraska", "Tennessee", "Wisconsin",
    "Alabama", "Kansas", "Idaho", "Kentucky", "Texas",
])

GROWTH_FEATURE_COLS = [
    "PRICE_CHANGE_YOY_3MO", "SALES_GROWTH_YOY_3MO", "PRICE_X_MATURITY",
]

# Elasticity of annual sales growth to store growth, applied directly (not via regression).
# Ramps from 0 at year 1 to full effect at year 5+.
STORE_GROWTH_ELASTICITY = 0.35

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_base_data():
    df_price = pd.read_csv("state_price_monthly.csv")
    df_disp  = pd.read_csv("num_dispensaries_per_month.csv")
    df_sales = pd.read_csv("state_sales_monthly.csv")

    df_price.columns = ["STATE", "SALES_MONTH", "STATE_PRICE"]
    df_disp.columns  = ["STATE", "SALES_MONTH", "COUNT_DISP"]
    df_sales.columns = ["STATE", "SALES_MONTH", "DOLLAR_SALES"]

    for df in [df_price, df_disp, df_sales]:
        df["SALES_MONTH"] = pd.to_datetime(df["SALES_MONTH"])

    df_disp = df_disp.sort_values(["STATE", "SALES_MONTH"])
    df_disp["NET_NEW_DISP"] = df_disp.groupby("STATE")["COUNT_DISP"].diff()

    df = (
        df_sales
        .merge(df_price, on=["STATE", "SALES_MONTH"], how="left")
        .merge(df_disp[["STATE", "SALES_MONTH", "NET_NEW_DISP", "COUNT_DISP"]],
               on=["STATE", "SALES_MONTH"], how="left")
    )

    df["LEGAL_DATE"] = pd.to_datetime(df["STATE"].map(LEGALIZATION_DATES))
    df["MONTHS_SINCE_LEGAL"] = (df["SALES_MONTH"] - df["LEGAL_DATE"]) / pd.Timedelta(days=30)
    df["YEARS_SINCE_LEGAL"]  = df["MONTHS_SINCE_LEGAL"] / 12

    df["COUNT_DISP_LAG1"]  = df.groupby("STATE")["COUNT_DISP"].shift(1)
    df["STORE_GROWTH_RATE"] = df["NET_NEW_DISP"] / df["COUNT_DISP_LAG1"]

    df["SALES_LAG12"]      = df.groupby("STATE")["DOLLAR_SALES"].shift(12)
    df["SALES_GROWTH_YOY"] = df["DOLLAR_SALES"] / df["SALES_LAG12"] - 1

    df["PRICE_LAG12"]      = df.groupby("STATE")["STATE_PRICE"].shift(12)
    df["PRICE_CHANGE_YOY"] = df["STATE_PRICE"] / df["PRICE_LAG12"] - 1

    df["SALES_GROWTH_YOY_3MO"] = (
        df.groupby("STATE")["SALES_GROWTH_YOY"]
        .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["PRICE_CHANGE_YOY_3MO"] = (
        df.groupby("STATE")["PRICE_CHANGE_YOY"]
        .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["PRICE_X_MATURITY"] = df["PRICE_CHANGE_YOY_3MO"] * df["YEARS_SINCE_LEGAL"]

    df["TARGET_GROWTH_YOY"] = (
        df.groupby("STATE")["DOLLAR_SALES"].shift(-1) /
        df.groupby("STATE")["DOLLAR_SALES"].shift(11)
    ) - 1

    df["rec"]       = df["STATE"].isin(REC_STATES).astype(int)
    df["years_rec"] = df["YEARS_SINCE_LEGAL"] * df["rec"]

    return df


@st.cache_data
def load_demo_data():
    df = pd.read_csv("base_segment_counts_per_state.csv")
    seg_cols = [c for c in df.columns if c != "state"]
    df["Total_pop"] = df[seg_cols].sum(axis=1)
    return df


@st.cache_data
def load_purchasing_data():
    return pd.read_csv("purchasing_beliefs.csv", index_col=0)


@st.cache_data
def load_category_data():
    # Always use the most recent category file
    files = sorted(glob.glob("get-monthly-sales-categories_*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[-1])
    df.columns = [c.strip() for c in df.columns]
    df["SALES_MONTH"] = pd.to_datetime(df["SALES_MONTH"])
    return df


@st.cache_data
def build_demos_processed(df_demos, df_purchasing):
    col = "RX/nonRX/health/beauty items:Stores bought pst 30 days Any marijuana dspnsry (A) Users/100 HHs"
    purchase_rates = df_purchasing[col].to_dict()
    seg_cols = [c for c in df_demos.columns if "Count" in c]

    demand = pd.Series(0.0, index=df_demos.index)
    for c in seg_cols:
        key = float(c.split()[0])
        if key in purchase_rates:
            demand += df_demos[c] * purchase_rates[key] / 100

    out = df_demos[["state", "Total_pop"]].copy()
    out["demo_demand_index"] = demand.values
    return out


@st.cache_data
def build_full_data(df_raw, df_demos_processed):
    df = df_raw.merge(df_demos_processed, left_on="STATE", right_on="state", how="left")
    df["disp_per_hh"]       = df["COUNT_DISP"] / df["Total_pop"]
    df["sales_per_capita"]  = df["DOLLAR_SALES"] / df["Total_pop"]
    df["TTM_SALES"] = (
        df.groupby("STATE")["DOLLAR_SALES"]
        .rolling(12, min_periods=12).sum().reset_index(level=0, drop=True)
    )
    df["sales_per_capita_ttm"] = df["TTM_SALES"] / df["Total_pop"]
    return df


# ── Model Training ────────────────────────────────────────────────────────────

@st.cache_data
def train_growth_model(df_raw):
    df = df_raw[df_raw["SALES_MONTH"] > pd.to_datetime("2023-01-01")].copy()
    df = df[df["TARGET_GROWTH_YOY"].between(-1, 2.5)]
    df = df[df["STATE"].isin(GROWTH_TRAINING_STATES)]
    df = df.dropna(subset=GROWTH_FEATURE_COLS + ["TARGET_GROWTH_YOY"])
    model = LinearRegression(fit_intercept=True)
    model.fit(df[GROWTH_FEATURE_COLS], df["TARGET_GROWTH_YOY"])
    return model


@st.cache_data
def train_structural_model(df_full):
    structural_train_states = [
        "Oregon", "Massachusetts", "Maryland", "Maine", "California", "North Dakota",
        "Utah", "Hawaii", "New Mexico", "Minnesota", "Montana", "Rhode Island",
        "Missouri", "Pennsylvania", "Michigan", "Colorado", "District of Columbia",
        "Virginia", "New York", "Connecticut", "Arizona", "Mississippi", "Ohio",
        "New Jersey", "Nevada", "Vermont", "Alaska", "South Dakota", "Arkansas",
        "West Virginia", "Illinois", "Oklahoma", "Washington", "Florida",
    ]
    df = df_full[
        (df_full["SALES_MONTH"] > pd.to_datetime("2023-01-01")) &
        df_full["TARGET_GROWTH_YOY"].between(-1, 2.5) &
        df_full["STATE"].isin(structural_train_states)
    ].copy()

    global_price_mean = df["STATE_PRICE"].mean()
    disp_mean         = df["disp_per_hh"].mean()

    df["trip_factor"]  = 1 / (1 + np.exp(-1.0 * (df["years_rec"] - 4.0)))
    df["disp_norm"]    = df["disp_per_hh"] / disp_mean
    df["access_factor"] = df["disp_norm"] / (1 + df["disp_norm"])
    df["price_factor"] = np.clip(df["STATE_PRICE"] / global_price_mean, 0.5, 1.5)
    df["pred_raw"]     = (
        df["demo_demand_index"] * df["trip_factor"] *
        df["access_factor"]    * df["price_factor"]
    )

    df = df.dropna(subset=["pred_raw", "sales_per_capita_ttm"])
    df = df[(df["pred_raw"] > 0) & (df["sales_per_capita_ttm"] > 0)]
    df["log_pred"]   = np.log(df["pred_raw"])
    df["log_actual"] = np.log(df["sales_per_capita_ttm"])

    model = LinearRegression()
    model.fit(df[["log_pred"]], df["log_actual"])
    return model, global_price_mean, disp_mean


# ── Projection Logic ──────────────────────────────────────────────────────────

def project_growth_model(df_state, growth_model, params):
    hist = df_state.sort_values("SALES_MONTH").copy()
    hist["TTM_SALES"] = hist["DOLLAR_SALES"].rolling(12, min_periods=12).sum()

    ttm_valid = hist["TTM_SALES"].dropna()
    if hist.empty or ttm_valid.empty:
        return hist, None

    latest    = hist.iloc[-1].copy()
    last_year = latest["SALES_MONTH"].year
    cur_ttm   = ttm_valid.iloc[-1]

    def _safe(val, fallback=0.0):
        return float(val) if pd.notna(val) else fallback

    cur_sales_growth = _safe(latest["SALES_GROWTH_YOY_3MO"])
    cur_store_growth = (
        params["store_growth_override"] if params["override_store_growth"]
        else _safe(latest["STORE_GROWTH_RATE"])
    )
    cur_price_change = (
        params["price_trend"] if params["override_price"]
        else _safe(latest["PRICE_CHANGE_YOY_3MO"])
    )
    cur_years = _safe(latest["YEARS_SINCE_LEGAL"], fallback=5.0)

    rows = []
    for i in range(1, params["years_forward"] + 1):
        cur_years += 1
        X = pd.DataFrame([{
            "PRICE_CHANGE_YOY_3MO": cur_price_change,
            "SALES_GROWTH_YOY_3MO": cur_sales_growth,
            "PRICE_X_MATURITY":     cur_price_change * cur_years,
        }])
        X = X.fillna(0)
        pred_growth = float(growth_model.predict(X[GROWTH_FEATURE_COLS])[0])

        # Direct store growth effect: ramps from 0 at year 1 to full at year 5+.
        # Ensures store growth is always neutral-to-positive, never a drag.
        maturity_weight = float(np.clip((cur_years - 1.0) / 4.0, 0.0, 1.0))
        pred_growth += max(0.0, cur_store_growth) * maturity_weight * STORE_GROWTH_ELASTICITY

        pred_growth = np.clip(pred_growth, params["growth_floor"], params["growth_cap"])
        cur_ttm     *= (1 + pred_growth)

        rows.append({
            "DATE": pd.Timestamp(f"{last_year + i}-01-01"),
            "PROJECTED_TTM_SALES": cur_ttm,
            "PRED_GROWTH": pred_growth,
        })
        cur_sales_growth  = pred_growth
        cur_store_growth *= params["store_decay"]
        cur_price_change *= params["price_decay"]

    return hist, pd.DataFrame(rows)


def project_structural_model(state, df_state, structural_model, global_price_mean,
                              disp_mean, df_demos_processed, params):
    midpoint   = params["scurve_midpoint"]
    steepness  = params["scurve_steepness"]
    adj_speed  = params["adjustment_speed"]

    has_history = df_state is not None and not df_state.empty

    if has_history:
        hist = df_state.sort_values("SALES_MONTH").copy()
        hist["TTM_TOTAL_SALES"] = (
            (hist["sales_per_capita"] * hist["Total_pop"])
            .rolling(12, min_periods=12).sum()
        )
        actual_ttm = hist["TTM_TOTAL_SALES"].dropna()
        has_history = not actual_ttm.empty

    if has_history:
        latest    = hist.iloc[-1].copy()
        last_year = latest["SALES_MONTH"].year
        cur_ttm   = actual_ttm.iloc[-1]

        avg_disp_growth  = (
            params["store_growth_override"] if params["override_store_growth"]
            else np.clip(hist["disp_per_hh"].pct_change().tail(12).mean(), -0.2, 0.3)
        )
        avg_price_growth = np.clip(hist["STATE_PRICE"].pct_change().tail(12).mean(), -0.2, 0.2)
        if pd.isna(avg_disp_growth):  avg_disp_growth  = 0.0
        if pd.isna(avg_price_growth): avg_price_growth = 0.0

        future = latest.copy()
    else:
        state_row = df_demos_processed[df_demos_processed["state"] == state]
        if state_row.empty:
            return None, None

        last_year = date.today().year
        cur_ttm   = 0.0
        avg_disp_growth  = 0.05
        avg_price_growth = -0.05
        future = pd.Series({
            "rec": 1, "years_rec": 0.0, "disp_per_hh": 0.0,
            "STATE_PRICE": global_price_mean,
            "demo_demand_index": float(state_row.iloc[0]["demo_demand_index"]),
            "Total_pop":         float(state_row.iloc[0]["Total_pop"]),
        })
        hist = None

    initial_disp  = float(future.get("disp_per_hh", 0.0))
    years_forward = max(params["years_forward"], 5) if not has_history else params["years_forward"]

    rows = []
    for i in range(1, years_forward + 1):
        if future.get("rec", 0) == 1:
            future["years_rec"] = float(future.get("years_rec", 0.0)) + 1
        else:
            future["years_rec"] = 0.0

        if has_history:
            future["disp_per_hh"]  *= (1 + avg_disp_growth)
            if params["override_price"]:
                future["STATE_PRICE"] *= (1 + params["price_trend"])
            else:
                future["STATE_PRICE"] *= (1 + avg_price_growth)
        else:
            ramp = 1 / (1 + np.exp(-0.8 * (i - midpoint)))
            future["disp_per_hh"]  = initial_disp + ramp * (disp_mean - initial_disp)
            future["STATE_PRICE"]  = float(future["STATE_PRICE"]) * 0.8 + global_price_mean * 0.2

        trip_factor   = 1 / (1 + np.exp(-steepness * (float(future["years_rec"]) - midpoint)))
        disp_norm     = float(future["disp_per_hh"]) / disp_mean
        access_factor = disp_norm / (1 + disp_norm)
        price_factor  = np.clip(float(future["STATE_PRICE"]) / global_price_mean, 0.5, 1.5)

        pred_raw = float(future["demo_demand_index"]) * trip_factor * access_factor * price_factor
        if pred_raw <= 0 or np.isnan(pred_raw):
            continue

        expected_pc    = np.exp(float(structural_model.predict([[np.log(pred_raw)]])[0]))
        expected_total = expected_pc * float(future["Total_pop"])

        projected_total = (
            cur_ttm + adj_speed * (expected_total - cur_ttm)
            if has_history else expected_total
        )

        rows.append({
            "DATE": pd.Timestamp(f"{last_year + i}-01-01"),
            "PROJECTED_TTM_TOTAL_SALES": projected_total,
            "EXPECTED_TTM_TOTAL_SALES":  expected_total,
            "trip_factor": trip_factor,
            "years_rec":   float(future["years_rec"]),
        })
        cur_ttm = projected_total

    return hist, pd.DataFrame(rows) if rows else None


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_dollars(x):
    """Format a dollar value as $2.6B or $450M."""
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    return f"${x/1e6:.0f}M"


# ── Chart Builders ────────────────────────────────────────────────────────────

# ── Hoodie Analytics brand palette ───────────────────────────────────────────
HOODIE_NAVY  = "#1B2D6B"   # primary navy
HOODIE_GREEN = "#5DC4A0"   # mint green
HOODIE_BLUE  = "#2E86AB"   # secondary blue (structural model)
HOODIE_LIGHT = "#A8D8EA"   # light blue (expected line)

COLORS = {
    "actual":     HOODIE_NAVY,
    "growth":     HOODIE_GREEN,
    "structural": HOODIE_BLUE,
    "expected":   HOODIE_LIGHT,
}

CATEGORY_COLORS = {
    "Flower":       HOODIE_GREEN,
    "Vapes":        HOODIE_NAVY,
    "Pre-Rolls":    "#F4A261",
    "Edibles":      "#9C6FBF",
    "Concentrates": "#E76F51",
    "Other":        "#9E9E9E",
}

CHART_FONT = dict(
    font_size=18,
    font_color=HOODIE_NAVY,
    xaxis_tickfont_size=16,
    xaxis_title_font_size=18,
    yaxis_tickfont_size=16,
    yaxis_title_font_size=18,
    legend_font_size=16,
    title_font_size=20,
    title_font_color=HOODIE_NAVY,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
)


def make_category_stacked_bar(state, df_cat, proj_g):
    """
    Stacked bar chart using 12-month rolling TTM sums.
    Historical: last TTM snapshot per year (2025+), so 2026 shows TTM-as-of-latest.
    Projected: growth model TTM × trailing category mix.
    All bars are on the same TTM scale — no gap between history and projections.
    """
    df_state = df_cat[df_cat["STATE"] == state].copy()
    if df_state.empty:
        return None

    df_state = df_state.sort_values(["CATEGORY_GROUP", "SALES_MONTH"])
    df_state["YEAR"] = df_state["SALES_MONTH"].dt.year
    categories = sorted(df_state["CATEGORY_GROUP"].unique())

    # ── TTM rolling sum per category ──────────────────────────────────────────
    df_state["TTM"] = (
        df_state.groupby("CATEGORY_GROUP")["TOTAL_DOLLARS"]
        .transform(lambda x: x.rolling(12, min_periods=12).sum())
    )

    # For each year >= 2025, take the last available TTM snapshot per category.
    # Complete years (Dec) give annual total = TTM ending Dec.
    # Partial current year gives TTM ending latest month (directly comparable to projections).
    annual = (
        df_state[df_state["TTM"].notna() & (df_state["YEAR"] >= 2025)]
        .groupby(["YEAR", "CATEGORY_GROUP"])["TTM"]
        .last()
        .reset_index()
    )
    if annual.empty:
        return None

    pivot = annual.pivot_table(
        index="YEAR", columns="CATEGORY_GROUP", values="TTM"
    ).fillna(0)

    # ── Category mix from the latest TTM snapshot (for projections) ───────────
    latest_ttm = df_state[df_state["TTM"].notna()].groupby("CATEGORY_GROUP")["TTM"].last()
    total_latest = latest_ttm.sum()
    cat_mix = (latest_ttm / total_latest).to_dict() if total_latest > 0 else {}

    # Label the most recent historical bar as partial if it doesn't end in Dec
    latest_month = df_state["SALES_MONTH"].max().month
    max_hist_year = pivot.index.max()
    is_partial = latest_month != 12

    fig = go.Figure()

    # ── Historical TTM bars ───────────────────────────────────────────────────
    for cat in categories:
        if cat not in pivot.columns:
            continue
        x_labels = [
            f"{yr} (TTM thru {df_state['SALES_MONTH'].max().strftime('%b')})" if (yr == max_hist_year and is_partial) else str(yr)
            for yr in pivot.index
        ]
        fig.add_trace(go.Bar(
            x=x_labels, y=pivot[cat].values,
            name=cat,
            marker_color=CATEGORY_COLORS.get(cat, "#607D8B"),
            legendgroup=cat,
            hovertemplate=f"<b>{cat}</b><br>%{{x}}: $%{{y:,.0f}}<extra>Historical TTM</extra>",
        ))

    # ── Projected TTM bars ────────────────────────────────────────────────────
    if proj_g is not None and not proj_g.empty and cat_mix:
        for _, row in proj_g.iterrows():
            year_str   = str(row["DATE"].year)
            total_proj = row["PROJECTED_TTM_SALES"]
            for cat in categories:
                share = cat_mix.get(cat, 0)
                fig.add_trace(go.Bar(
                    x=[year_str], y=[total_proj * share],
                    name=cat,
                    marker_color=CATEGORY_COLORS.get(cat, "#607D8B"),
                    marker_pattern_shape="/",
                    legendgroup=cat,
                    showlegend=False,
                    hovertemplate=f"<b>{cat}</b><br>%{{x}} (proj TTM): $%{{y:,.0f}}<extra>Projected</extra>",
                ))

    fig.update_layout(
        barmode="stack",
        title=f"{state} — TTM Sales by Category  *(hatched = projected)*",
        xaxis_title="Year",
        yaxis_title="TTM Sales",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, font=dict(size=16)),
        hovermode="x unified",
        height=460,
        template="plotly_white",
        xaxis_gridcolor="#E8EDF5",
        yaxis_gridcolor="#E8EDF5",
        margin=dict(t=60, b=160),
        **CHART_FONT,
    )
    return fig


def make_structural_bar_chart(state, hist_s, proj_s):
    """
    Bar chart for structural model states.
    - Legal states: historical TTM bars (2025+), a visual gap, then hatched projected bars.
    - Unlegal states: no history; projected bars labeled 'Year 1', 'Year 2', ...
      with x-axis title 'Years since full legalization'.
    """
    fig = go.Figure()

    # Determine whether this state has a real market to show historically
    has_hist = (
        state not in NO_MARKET_STATES and
        hist_s is not None and not hist_s.empty and
        "TTM_TOTAL_SALES" in hist_s.columns and
        hist_s["TTM_TOTAL_SALES"].notna().any()
    )

    if has_hist:
        # ── Historical TTM bars ───────────────────────────────────────────────
        h = hist_s.copy()
        h["YEAR"] = h["SALES_MONTH"].dt.year
        h = h[h["TTM_TOTAL_SALES"].notna() & (h["YEAR"] >= 2025)]
        annual = h.groupby("YEAR")["TTM_TOTAL_SALES"].last().reset_index()

        latest_month = hist_s["SALES_MONTH"].max().month
        max_yr = annual["YEAR"].max()
        x_hist = [
            f"{yr} (TTM thru {hist_s['SALES_MONTH'].max().strftime('%b')})"
            if (yr == max_yr and latest_month != 12) else str(yr)
            for yr in annual["YEAR"]
        ]
        fig.add_trace(go.Bar(
            x=x_hist, y=annual["TTM_TOTAL_SALES"],
            name="Historical TTM",
            marker_color=COLORS["actual"],
            hovertemplate="<b>%{x}</b>: $%{y:,.0f}<extra>Historical TTM</extra>",
        ))

        # Invisible spacer bar — creates a visual gap before projected bars
        fig.add_trace(go.Bar(
            x=["  "], y=[0],
            showlegend=False,
            marker_color="rgba(0,0,0,0)",
            hoverinfo="skip",
        ))

        # ── Projected bars (calendar years) ──────────────────────────────────
        if proj_s is not None and not proj_s.empty:
            x_proj = proj_s["DATE"].dt.year.astype(str).tolist()
            fig.add_trace(go.Bar(
                x=x_proj, y=proj_s["PROJECTED_TTM_TOTAL_SALES"],
                name="Projected",
                marker_color=COLORS["structural"],
                marker_pattern_shape="/",
                hovertemplate="<b>%{x} (proj TTM)</b>: $%{y:,.0f}<extra>Projected</extra>",
            ))

        xaxis_title = "Year"

    else:
        # ── Unlegal state: projected only, labeled Year 1 / Year 2 / … ───────
        if proj_s is not None and not proj_s.empty:
            x_proj = [f"Year {i+1}" for i in range(len(proj_s))]
            fig.add_trace(go.Bar(
                x=x_proj, y=proj_s["PROJECTED_TTM_TOTAL_SALES"],
                name="Projected",
                marker_color=COLORS["structural"],
                marker_pattern_shape="/",
                hovertemplate="<b>%{x} (proj TTM)</b>: $%{y:,.0f}<extra>Projected</extra>",
            ))

        xaxis_title = "Years since full legalization"

    fig.update_layout(
        barmode="relative",
        title=f"{state} — TTM Sales Projection  *(hatched = projected)*",
        xaxis_title=xaxis_title,
        yaxis_title="TTM Sales",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, font=dict(size=16)),
        hovermode="x unified",
        height=460,
        template="plotly_white",
        xaxis_gridcolor="#E8EDF5",
        yaxis_gridcolor="#E8EDF5",
        margin=dict(t=60, b=160),
        **CHART_FONT,
    )
    return fig


def make_state_chart(state, hist_g, proj_g, hist_s, proj_s, model_choice):
    fig = go.Figure()
    show_growth = model_choice in ("Market Trend Model", "Both")
    show_struct = model_choice in ("Structural Demographic Model", "Both")

    # Actual TTM
    actual_hist = hist_g if (show_growth and hist_g is not None) else (hist_s if show_struct else None)
    actual_col  = "TTM_SALES" if actual_hist is hist_g else "TTM_TOTAL_SALES"
    if actual_hist is not None:
        ttm = actual_hist[actual_col].dropna()
        if not ttm.empty:
            fig.add_trace(go.Scatter(
                x=actual_hist.loc[ttm.index, "SALES_MONTH"], y=ttm,
                name="Actual (TTM)", mode="lines+markers",
                line=dict(color=COLORS["actual"], width=2),
                hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Actual</extra>",
            ))

    if show_growth and proj_g is not None and not proj_g.empty:
        fig.add_trace(go.Scatter(
            x=proj_g["DATE"], y=proj_g["PROJECTED_TTM_SALES"],
            name="Market Trend Model", mode="lines+markers",
            line=dict(color=COLORS["growth"], width=2, dash="dash"),
            hovertemplate="%{x|%Y}: $%{y:,.0f}<extra>Growth Model</extra>",
        ))

    if show_struct and proj_s is not None and not proj_s.empty:
        fig.add_trace(go.Scatter(
            x=proj_s["DATE"], y=proj_s["PROJECTED_TTM_TOTAL_SALES"],
            name="Structural Demographic Model", mode="lines+markers",
            line=dict(color=COLORS["structural"], width=2, dash="dot"),
            hovertemplate="%{x|%Y}: $%{y:,.0f}<extra>Structural</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=proj_s["DATE"], y=proj_s["EXPECTED_TTM_TOTAL_SALES"],
            name="Structural Demographic Expected", mode="lines",
            line=dict(color=COLORS["expected"], width=1, dash="dot"),
            opacity=0.6,
            hovertemplate="%{x|%Y}: $%{y:,.0f}<extra>Expected</extra>",
        ))

    fig.update_layout(
        title=f"{state} — TTM Sales Projection",
        xaxis_title="Date", yaxis_title="TTM Sales",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, font=dict(size=16)),
        hovermode="x unified", height=460, template="plotly_white",
        margin=dict(t=60, b=160),
        **CHART_FONT,
    )
    return fig


def make_all_states_chart(df_summary, year_col):
    df_plot = df_summary[df_summary[year_col].notna()].sort_values(year_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_plot[year_col] / 1e6,
        y=df_plot["State"],
        orientation="h",
        marker_color=COLORS["growth"],
        hovertemplate="%{y}: $%{x:,.0f}M<extra></extra>",
    ))
    fig.update_layout(
        title=f"Projected TTM Sales by State ({year_col})",
        xaxis_title="TTM Sales ($M)", yaxis_title="",
        height=max(400, len(df_plot) * 22),
        template="plotly_white",
        xaxis_gridcolor="#E8EDF5",
        yaxis_gridcolor="#E8EDF5", margin=dict(l=160),
        **CHART_FONT,
    )
    return fig


# ── Sidebar defaults (per-state, data-driven) ────────────────────────────────

@st.cache_data
def compute_all_state_defaults(df_raw):
    """
    For each state, compute trailing 12-month average price change and store
    growth rate. Falls back to national average for states with no data.
    Returns dict: state -> {price_trend_pct, store_growth_pct}.
    """
    cutoff = df_raw["SALES_MONTH"].max() - pd.DateOffset(months=12)
    recent = df_raw[df_raw["SALES_MONTH"] > cutoff]

    national_price = recent["PRICE_CHANGE_YOY"].mean()
    national_store = recent["STORE_GROWTH_RATE"].mean()

    result = {}
    for state in ALL_MODEL_STATES:
        rows = recent[recent["STATE"] == state]
        price = rows["PRICE_CHANGE_YOY"].mean()  if not rows.empty else national_price
        store = rows["STORE_GROWTH_RATE"].mean() if not rows.empty else national_store
        if pd.isna(price): price = national_price
        if pd.isna(store): store = national_store
        result[state] = {
            "price_trend_pct":  int(np.clip(round(price * 100), -30, 20)),
            "store_growth_pct": int(np.clip(round(store * 100), -20, 50)),
        }
    return result


def _state_defaults(state, all_state_defaults):
    """Full defaults dict for a given state."""
    sd = all_state_defaults[state]
    return {
        "years_forward":    5,
        "price_trend_pct":  sd["price_trend_pct"],
        "store_growth_pct": sd["store_growth_pct"],
        # advanced
        "store_decay":      0.90,
        "price_decay":      0.95,
        "adjustment_speed": 0.40,
        "scurve_midpoint":  4.0,
        "scurve_steepness": 1.0,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(all_state_defaults):
    with st.sidebar:
        st.header("Settings")

        state = st.selectbox("State", ALL_MODEL_STATES,
                             index=ALL_MODEL_STATES.index("New York"))

        # When state changes, reset sliders to that state's historical values
        if st.session_state.get("_active_state") != state:
            for k, v in _state_defaults(state, all_state_defaults).items():
                st.session_state[k] = v
            st.session_state["_active_state"] = state

        auto_model = "Market Trend Model" if state in STATES_USE_GROWTH_MODEL else "Structural Demographic Model"
        model_choice = st.radio("Model", ["Auto", "Market Trend Model", "Structural Demographic Model", "Both"])
        if model_choice == "Auto":
            st.caption(f"Auto-selected: **{auto_model}**")
            model_choice = auto_model

        st.divider()

        if st.button("↺ Reset to state defaults", use_container_width=True):
            for k, v in _state_defaults(state, all_state_defaults).items():
                st.session_state[k] = v
            st.rerun()

        sd = all_state_defaults[state]
        st.caption(
            f"State defaults — price: **{sd['price_trend_pct']:+d}%** · "
            f"store growth: **{sd['store_growth_pct']:+d}%**"
        )

        st.subheader("Projection")
        years_forward = st.slider("Years forward", 1, 10, key="years_forward")

        st.subheader("Price Trend")
        price_trend = st.slider(
            "Current annual price change", -30, 20,
            format="%d%%",
            key="price_trend_pct",
            help="Sets today's price trend. Effect decays by 5% each year, converging toward flat pricing over the projection window.",
        ) / 100

        st.subheader("Store / Dispensary Growth")
        store_growth_override = st.slider(
            "Current annual store growth", -20, 50,
            format="%d%%",
            key="store_growth_pct",
            help="Sets today's store growth rate. Effect decays by 10% each year, reflecting a natural slowdown in new store openings over time.",
        ) / 100

        store_decay      = 0.90
        price_decay      = 0.95
        adjustment_speed = 0.40
        scurve_midpoint  = 4.0
        scurve_steepness = 1.0

    params = dict(
        years_forward=years_forward,
        price_trend=price_trend, override_price=True,
        store_growth_override=store_growth_override, override_store_growth=True,
        store_decay=store_decay, price_decay=price_decay,
        growth_floor=-0.20, growth_cap=0.30,
        adjustment_speed=adjustment_speed,
        scurve_midpoint=scurve_midpoint, scurve_steepness=scurve_steepness,
    )
    return state, model_choice, params


# ── All-States Computation (cached on params) ─────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_all_states(
    _df_raw, _df_full, _df_demos_processed,
    _growth_model, _structural_model,
    global_price_mean, disp_mean,
    params_key,   # hashable tuple — used as cache key
):
    params = dict(params_key)
    rows = []
    for s in ALL_MODEL_STATES:
        df_s_raw  = _df_raw[_df_raw["STATE"] == s]
        df_s_full = _df_full[_df_full["STATE"] == s]

        _, pg = project_growth_model(df_s_raw, _growth_model, params)
        _, ps = project_structural_model(
            s, df_s_full if not df_s_full.empty else None,
            _structural_model, global_price_mean, disp_mean, _df_demos_processed, params,
        )

        gv = float(pg.iloc[-1]["PROJECTED_TTM_SALES"])    if pg is not None and not pg.empty else None
        sv = float(ps.iloc[-1]["PROJECTED_TTM_TOTAL_SALES"]) if ps is not None and not ps.empty else None
        auto = "Market Trend Model" if s in STATES_USE_GROWTH_MODEL else "Structural Demographic Model"
        mv   = gv if auto == "Market Trend Model" else sv

        rows.append({"State": s, "Model": auto,
                     "Market Trend Model": gv, "Structural Demographic Model": sv, "Selected": mv})
    return pd.DataFrame(rows)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Global CSS / brand styling ────────────────────────────────────────────
    st.markdown("""
        <style>
        /* ── Page background ── */
        .stApp { background-color: #FFFFFF; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: #F0F5F9;
            border-right: 2px solid #1B2D6B22;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown p {
            color: #1B2D6B !important;
        }

        /* ── Page header ── */
        .hoodie-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding-bottom: 6px;
            border-bottom: 3px solid #5DC4A0;
            margin-bottom: 18px;
        }
        .hoodie-header h1 {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1B2D6B;
            margin: 0;
        }
        .hoodie-header span.sub {
            font-size: 0.95rem;
            color: #5DC4A0;
            font-weight: 500;
        }

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {
            background: #F0F5F9;
            border: 1px solid #1B2D6B22;
            border-left: 4px solid #5DC4A0;
            border-radius: 8px;
            padding: 10px 16px;
        }
        div[data-testid="metric-container"] label {
            color: #1B2D6B !important;
            font-weight: 600;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #1B2D6B !important;
            font-size: 1.6rem;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 10px 28px;
            color: #1B2D6B;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #5DC4A0 !important;
            color: #1B2D6B !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background-color: #1B2D6B;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #5DC4A0;
            color: white;
        }

        /* ── Download button ── */
        .stDownloadButton > button {
            background-color: #5DC4A0;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
        }

        /* ── Expander header ── */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #1B2D6B !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Branded header ────────────────────────────────────────────────────────
    st.markdown("""
        <div class="hoodie-header">
            <div>
                <h1>Cannabis Market Estimates</h1>
                <span class="sub">Powered by Hoodie Analytics</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data & training models…"):
        df_raw            = load_base_data()
        df_demos_raw      = load_demo_data()
        df_purchasing     = load_purchasing_data()
        df_cat            = load_category_data()
        df_demos_processed = build_demos_processed(df_demos_raw, df_purchasing)
        df_full           = build_full_data(df_raw, df_demos_processed)
        growth_model      = train_growth_model(df_raw)
        structural_model, global_price_mean, disp_mean = train_structural_model(df_full)

    all_state_defaults = compute_all_state_defaults(df_raw)

    state, model_choice, params = render_sidebar(all_state_defaults)
    tab_state, tab_all = st.tabs(["State View", "All States"])

    # ── Tab 1: Single State ───────────────────────────────────────────────────
    with tab_state:
        df_s_raw  = df_raw[df_raw["STATE"] == state]
        df_s_full = df_full[df_full["STATE"] == state]

        hist_g, proj_g = project_growth_model(df_s_raw, growth_model, params)
        hist_s, proj_s = project_structural_model(
            state, df_s_full if not df_s_full.empty else None,
            structural_model, global_price_mean, disp_mean, df_demos_processed, params,
        )

        # KPI cards
        final_year = date.today().year + params["years_forward"]
        if model_choice == "Both":
            c1, c2, c3 = st.columns(3)
            if proj_g is not None and not proj_g.empty:
                c1.metric(f"Market Trend {final_year}", fmt_dollars(proj_g.iloc[-1]["PROJECTED_TTM_SALES"]))
            if proj_s is not None and not proj_s.empty:
                c2.metric(f"Structural Demographic {final_year}", fmt_dollars(proj_s.iloc[-1]["PROJECTED_TTM_TOTAL_SALES"]))
            if proj_g is not None and not proj_g.empty:
                c3.metric("Avg proj'd growth", f"{proj_g['PRED_GROWTH'].mean():.1%}")
        elif model_choice == "Market Trend Model" and proj_g is not None and not proj_g.empty:
            c1, c2 = st.columns(2)
            c1.metric(f"Market Trend {final_year}", fmt_dollars(proj_g.iloc[-1]["PROJECTED_TTM_SALES"]))
            c2.metric("Avg proj'd growth", f"{proj_g['PRED_GROWTH'].mean():.1%}")
        elif model_choice == "Structural Demographic Model" and proj_s is not None and not proj_s.empty:
            ttm_vals = proj_s["PROJECTED_TTM_TOTAL_SALES"]
            growth = ttm_vals.pct_change().mean()
            c1, c2 = st.columns(2)
            c1.metric(f"Structural Demographic {final_year}", fmt_dollars(ttm_vals.iloc[-1]))
            c2.metric("Avg proj'd growth", f"{growth:.1%}" if pd.notna(growth) else "—")

        # ── Primary chart ─────────────────────────────────────────────────────
        if model_choice == "Market Trend Model" and df_cat is not None:
            cat_fig = make_category_stacked_bar(state, df_cat, proj_g)
            st.plotly_chart(cat_fig if cat_fig is not None else
                            make_structural_bar_chart(state, hist_g, proj_g),
                            use_container_width=True)
        elif model_choice == "Structural Demographic Model":
            st.plotly_chart(make_structural_bar_chart(state, hist_s, proj_s),
                            use_container_width=True)
        elif model_choice == "Both":
            cat_fig = make_category_stacked_bar(state, df_cat, proj_g) if df_cat is not None else None
            fig_left  = cat_fig if cat_fig is not None else make_structural_bar_chart(state, hist_g, proj_g)
            fig_right = make_structural_bar_chart(state, hist_s, proj_s)

            # Shared y-axis: find the max stacked bar height across both figures
            def _fig_ymax(fig):
                if fig is None:
                    return 0
                from collections import defaultdict
                stacks = defaultdict(float)
                for trace in fig.data:
                    xs = trace.x if trace.x is not None else []
                    ys = trace.y if trace.y is not None else []
                    for x, y in zip(xs, ys):
                        try:
                            v = float(y)
                            if v == v:  # exclude NaN
                                stacks[x] += v
                        except (TypeError, ValueError):
                            pass
                return max(stacks.values()) if stacks else 0

            shared_max = max(_fig_ymax(fig_left), _fig_ymax(fig_right)) * 1.05
            if shared_max > 0:
                for fig in [fig_left, fig_right]:
                    if fig is not None:
                        fig.update_layout(yaxis_range=[0, shared_max])

            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(fig_left, use_container_width=True)
            with c_right:
                st.plotly_chart(fig_right, use_container_width=True)

        with st.expander("Projection tables"):
            col_a, col_b = st.columns(2)
            if proj_g is not None and not proj_g.empty:
                with col_a:
                    st.write("**Market Trend Model**")
                    st.dataframe(
                        proj_g.assign(
                            DATE=proj_g["DATE"].dt.year,
                            PROJECTED_TTM_SALES=proj_g["PROJECTED_TTM_SALES"].map("${:,.0f}".format),
                            PRED_GROWTH=proj_g["PRED_GROWTH"].map("{:.1%}".format),
                        ),
                        hide_index=True,
                    )
            if proj_s is not None and not proj_s.empty:
                with col_b:
                    st.write("**Structural Demographic Model**")
                    df_disp = proj_s[["DATE", "PROJECTED_TTM_TOTAL_SALES"]].copy()
                    df_disp["GROWTH_RATE"] = df_disp["PROJECTED_TTM_TOTAL_SALES"].pct_change()
                    df_disp["DATE"] = df_disp["DATE"].dt.year
                    df_disp["PROJECTED_TTM_TOTAL_SALES"] = df_disp["PROJECTED_TTM_TOTAL_SALES"].map("${:,.0f}".format)
                    df_disp["GROWTH_RATE"] = df_disp["GROWTH_RATE"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                    st.dataframe(df_disp.rename(columns={
                        "DATE": "Year",
                        "PROJECTED_TTM_TOTAL_SALES": "Projected TTM Sales",
                        "GROWTH_RATE": "YoY Growth",
                    }), hide_index=True)

    # ── Tab 2: All States ─────────────────────────────────────────────────────
    with tab_all:
        st.subheader("All States Summary")

        params_key = tuple(sorted(params.items()))
        with st.spinner("Computing projections for all states…"):
            df_summary = compute_all_states(
                df_raw, df_full, df_demos_processed,
                growth_model, structural_model,
                global_price_mean, disp_mean,
                params_key,
            )

        final_year = date.today().year + params["years_forward"]

        total = df_summary["Selected"].sum(skipna=True)
        st.metric(f"Total US Market — {final_year}", f"${total/1e9:.1f}B")

        st.plotly_chart(
            make_all_states_chart(df_summary, "Selected"),
            use_container_width=True,
        )

        df_display = df_summary.sort_values("Selected", ascending=False, na_position="last").copy()
        for c in ["Market Trend Model", "Structural Demographic Model", "Selected"]:
            df_display[c] = df_display[c].apply(fmt_dollars)
        df_display = df_display.rename(columns={
            "Market Trend Model": f"Market Trend ({final_year})",
            "Structural Demographic Model": f"Structural Demographic ({final_year})",
            "Selected":     f"Selected ({final_year})",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        csv = df_summary.to_csv(index=False).encode()
        st.download_button("Download projections CSV", csv,
                           f"market_projections_{date.today()}.csv", "text/csv")



if __name__ == "__main__":
    main()
