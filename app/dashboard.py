"""
Toyota Canarias — Dealership Intelligence Dashboard
Streamlit page powered by src/analytics.py + Plotly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.analytics import (
    load_logs,
    kpi_summary,
    fuel_distribution,
    body_type_distribution,
    intent_distribution,
    sentiment_distribution,
    model_distribution,
    use_case_distribution,
    budget_bins,
    fuel_body_cooccurrence,
    session_stats,
    queries_by_day,
    queries_by_hour,
    unmet_demand,
)

# ── Toyota brand colours ──────────────────────────────────────────────────────
TOYOTA_RED  = "#EB0A1E"
TOYOTA_DARK = "#1A1A1A"
TOYOTA_GREY = "#2C2C2C"
TOYOTA_LIGHT= "#F5F5F5"

PALETTE = [
    "#EB0A1E", "#FF6B6B", "#FF9E9E",
    "#C0392B", "#922B21", "#641E16",
    "#FF4D5E", "#E74C3C",
]

PLOTLY_TEMPLATE = "plotly_dark"


def _fig_layout(fig, title: str = ""):
    fig.update_layout(
        paper_bgcolor=TOYOTA_GREY,
        plot_bgcolor=TOYOTA_GREY,
        font=dict(color="#FFFFFF", size=13),
        title=dict(text=title, font=dict(color=TOYOTA_RED, size=16)),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def _empty_msg(label: str):
    st.info(f"No data available for **{label}** yet.")


# ── KPI cards ─────────────────────────────────────────────────────────────────

def render_kpis(df):
    kpis = kpi_summary(df)
    if not kpis:
        st.warning("No log data found. Run the chatbot to generate data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    card_style = (
        "background:#2C2C2C;border-left:4px solid #EB0A1E;"
        "border-radius:8px;padding:16px 20px;margin-bottom:8px;"
    )

    def kpi_card(col, icon, label, value):
        col.markdown(
            f"""<div style="{card_style}">
                <div style="color:#aaa;font-size:0.78rem;margin-bottom:4px">{icon} {label}</div>
                <div style="color:#fff;font-size:1.7rem;font-weight:800">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    kpi_card(c1, "💬", "Total Queries",    kpis["total_queries"])
    kpi_card(c2, "👥", "Unique Sessions",  kpis["unique_sessions"])
    budget_val = (
        f"€{int(kpis['avg_budget']):,}" if kpis.get("avg_budget") else "N/A"
    )
    kpi_card(c3, "💶", "Avg Budget",       budget_val)
    kpi_card(c4, "⛽", "Top Fuel Type",    kpis["top_fuel"])


# ── individual chart renderers ────────────────────────────────────────────────

def render_fuel(df):
    data = fuel_distribution(df)
    if data.empty:
        return _empty_msg("Fuel Preference")
    fig = px.bar(
        x=data.index.str.title(), y=data.values,
        labels={"x": "Fuel Type", "y": "Queries"},
        color=data.index,
        color_discrete_sequence=PALETTE,
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Fuel Type Demand")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_body(df):
    data = body_type_distribution(df)
    if data.empty:
        return _empty_msg("Body Type")
    fig = px.bar(
        x=data.index.str.title(), y=data.values,
        labels={"x": "Body Type", "y": "Queries"},
        color=data.index,
        color_discrete_sequence=PALETTE,
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Body Type Demand")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_intent(df):
    data = intent_distribution(df)
    if data.empty:
        return _empty_msg("Intent Breakdown")
    fig = px.pie(
        names=data.index.str.title(), values=data.values,
        color_discrete_sequence=PALETTE,
        template=PLOTLY_TEMPLATE,
        hole=0.35,
    )
    fig = _fig_layout(fig, "Intent Breakdown")
    fig.update_traces(textfont_color="#fff")
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment(df):
    data = sentiment_distribution(df)
    if data.empty:
        return _empty_msg("Sentiment")
    SENT_COLOURS = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
    colours = [SENT_COLOURS.get(k, TOYOTA_RED) for k in data.index]
    fig = px.pie(
        names=data.index.str.title(), values=data.values,
        color_discrete_sequence=colours,
        template=PLOTLY_TEMPLATE,
        hole=0.35,
    )
    fig = _fig_layout(fig, "Sentiment Distribution")
    fig.update_traces(textfont_color="#fff")
    st.plotly_chart(fig, use_container_width=True)


def render_models(df):
    data = model_distribution(df).head(10)
    if data.empty:
        return _empty_msg("Toyota Model Demand")
    fig = px.bar(
        x=data.values, y=data.index.str.title(),
        orientation="h",
        labels={"x": "Queries", "y": "Model"},
        color=data.values,
        color_continuous_scale=["#922B21", TOYOTA_RED, "#FF6B6B"],
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Most Requested Toyota Models")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_use_cases(df):
    data = use_case_distribution(df).head(8)
    if data.empty:
        return _empty_msg("Use Cases")
    fig = px.bar(
        x=data.values, y=data.index.str.title(),
        orientation="h",
        labels={"x": "Queries", "y": "Use Case"},
        color=data.values,
        color_continuous_scale=["#641E16", TOYOTA_RED, "#FF9E9E"],
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Top Customer Use Cases")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_budget(df):
    data = budget_bins(df)
    if data.empty:
        return _empty_msg("Budget Distribution")
    fig = px.bar(
        x=data.index.astype(str), y=data.values,
        labels={"x": "Budget Range", "y": "Queries"},
        color=data.values,
        color_continuous_scale=["#641E16", TOYOTA_RED, "#FF9E9E"],
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Budget Distribution")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence(df):
    matrix = fuel_body_cooccurrence(df)
    if matrix.empty:
        return _empty_msg("Fuel × Body Co-occurrence")
    fig = px.imshow(
        matrix,
        color_continuous_scale=["#1A1A1A", "#641E16", TOYOTA_RED, "#FF9E9E"],
        template=PLOTLY_TEMPLATE,
        text_auto=True,
        aspect="auto",
    )
    fig = _fig_layout(fig, "Fuel Type × Body Type Co-occurrence")
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_timeseries(df):
    daily = queries_by_day(df)
    if daily.empty:
        return _empty_msg("Query Volume Over Time")
    ts_df = daily.reset_index()
    ts_df.columns = ["date", "count"]
    ts_df["date"] = pd.to_datetime(ts_df["date"])
    fig = px.line(
        ts_df, x="date", y="count",
        labels={"date": "Date", "count": "Queries"},
        template=PLOTLY_TEMPLATE,
        markers=True,
    )
    fig.update_traces(line_color=TOYOTA_RED, marker_color="#FF6B6B")
    fig = _fig_layout(fig, "Query Volume Over Time")
    st.plotly_chart(fig, use_container_width=True)


def render_hourly(df):
    hourly = queries_by_hour(df)
    if hourly.empty:
        return _empty_msg("Hourly Activity")
    h_df = hourly.reset_index()
    h_df.columns = ["hour", "count"]
    fig = px.bar(
        h_df, x="hour", y="count",
        labels={"hour": "Hour of Day", "count": "Queries"},
        color="count",
        color_continuous_scale=["#641E16", TOYOTA_RED, "#FF9E9E"],
        template=PLOTLY_TEMPLATE,
    )
    fig = _fig_layout(fig, "Queries by Hour of Day")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_unmet(df):
    unmet = unmet_demand(df)
    if unmet.empty:
        st.success("No unmet demand signals detected — great inventory coverage!")
        return
    st.markdown(
        f"<span style='color:{TOYOTA_RED};font-weight:700'>"
        f"⚠ {len(unmet)} queries with expressed preferences but no matching model</span>",
        unsafe_allow_html=True,
    )
    # friendly display names
    rename_map = {
        "timestamp": "Time",
        "session_id": "Session",
        "user_message": "Customer Message",
        "intent": "Intent",
        "fuel_preference": "Fuel",
        "body_type": "Body",
        "budget_mentioned": "Budget",
        "use_case": "Use Case",
    }
    display = unmet.rename(columns={k: v for k, v in rename_map.items() if k in unmet.columns})
    if "Customer Message" in display.columns:
        display["Customer Message"] = display["Customer Message"].str[:80] + "…"
    st.dataframe(display, use_container_width=True, height=320)


# ── main render ───────────────────────────────────────────────────────────────

def render_dashboard():
    # ── CSS overrides ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .stApp { background-color: #0f0f0f; }
        h1, h2, h3 { color: #EB0A1E !important; }
        .section-title {
            font-size: 1.1rem; font-weight: 700; color: #EB0A1E;
            margin: 1.5rem 0 0.5rem 0; letter-spacing: 0.5px;
        }
        .stDataFrame { background: #2C2C2C; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Dealership Intelligence")
    st.caption("Real-time demand analysis from Sofia's customer conversations")
    st.divider()

    # ── load data ─────────────────────────────────────────────────────────────
    df = load_logs()

    if df.empty:
        st.warning(
            "No query logs found at `query_logs/query_log.csv`. "
            "Start chatting with Sofia to generate data!"
        )
        return

    # ── date filter ───────────────────────────────────────────────────────────
    if "date" in df.columns and df["date"].notna().any():
        min_d = pd.to_datetime(df["date"].min())
        max_d = pd.to_datetime(df["date"].max())
        with st.expander("🗓 Filter by date range", expanded=False):
            col_a, col_b = st.columns(2)
            start = col_a.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
            end   = col_b.date_input("To",   value=max_d, min_value=min_d, max_value=max_d)
        df = df[(pd.to_datetime(df["date"]) >= pd.to_datetime(start)) &
                (pd.to_datetime(df["date"]) <= pd.to_datetime(end))]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    render_kpis(df)
    st.divider()

    # ── row 1: fuel & body ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Demand Breakdown</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        render_fuel(df)
    with c2:
        render_body(df)

    # ── row 2: intent & sentiment ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Conversation Insights</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        render_intent(df)
    with c4:
        render_sentiment(df)

    # ── row 3: models & use cases ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Product & Use Case Signals</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        render_models(df)
    with c6:
        render_use_cases(df)

    # ── row 4: budget ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Budget Distribution</div>', unsafe_allow_html=True)
    render_budget(df)

    # ── row 5: co-occurrence ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Fuel × Body Co-occurrence Matrix</div>', unsafe_allow_html=True)
    render_cooccurrence(df)

    # ── row 6: time series + hourly ───────────────────────────────────────────
    st.markdown('<div class="section-title">Temporal Patterns</div>', unsafe_allow_html=True)
    render_timeseries(df)
    render_hourly(df)

    # ── row 7: session stats ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Session Analysis</div>', unsafe_allow_html=True)
    stats = session_stats(df)
    if stats:
        sc1, sc2 = st.columns(2)
        sc1.metric("Avg Queries / Session", stats["avg_queries_per_session"])
        sc2.metric("Max Queries in One Session", stats["max_queries"])
        st.caption("Top 5 Most Active Sessions")
        if "top_sessions" in stats:
            st.dataframe(stats["top_sessions"], use_container_width=True, hide_index=True)

    # ── row 8: unmet demand ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Unmet Demand Signals</div>', unsafe_allow_html=True)
    st.caption("Queries where customers expressed preferences but no Toyota model was matched — potential stock or knowledge gaps.")
    render_unmet(df)

    st.divider()
    st.caption(f"Dashboard powered by {len(df):,} conversation records · Toyota Canarias BI")
