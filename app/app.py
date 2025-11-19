# app/app.py

from pathlib import Path
import random
import pandas as pd
import joblib
import altair as alt
import streamlit as st

# ---------- 1. Locate project paths ----------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATTERNS = [
    "salary_data_cleaned.csv",
    "prediction_salary.csv",
    "prediction salary.csv",
    "prediction-salary.csv",
    "predictionsalary.csv",
    "*predict*salary*.csv",
    "*salary*.csv",
]
MODEL_PATTERNS = [
    "salary_model.pkl",
    "model.pkl",
]

def find_first(pattern: str):
    """Return first file under BASE_DIR matching pattern, or None."""
    matches = [p for p in BASE_DIR.rglob(pattern)]
    return matches[0] if matches else None

def find_resource(patterns: list[str], default_pattern: str | None = None):
    """Try multiple filename patterns to locate a resource."""
    for pattern in patterns:
        path = find_first(pattern)
        if path:
            return path
    if default_pattern:
        return find_first(default_pattern)
    return None

DATA_PATH = find_resource(DATASET_PATTERNS)
if DATA_PATH is None:
    # fall back to the first CSV we can find (excluding virtual environments)
    csv_candidates = [
        p
        for p in BASE_DIR.rglob("*.csv")
        if "venv" not in p.parts and ".git" not in p.parts
    ]
    if csv_candidates:
        DATA_PATH = csv_candidates[0]
MODEL_PATH = find_resource(MODEL_PATTERNS, "*.pkl")

st.set_page_config(page_title="Job Salary Prediction", page_icon="💼", layout="wide")

CUSTOM_CSS = """
<style>
:root {
    --bg-dark: #020817;
    --bg-card: rgba(15, 23, 42, 0.78);
    --glass: rgba(15, 118, 110, 0.08);
    --accent: #38bdf8;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #155e75, #020617 55%);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(2, 8, 23, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.2);
}
.hero {
    border-radius: 32px;
    padding: 32px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, rgba(51,65,92,.9), rgba(8,47,73,.8));
    border: 1px solid rgba(94, 234, 212, 0.25);
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.45);
    position: relative;
    overflow: hidden;
}
.hero:before {
    content: "";
    position: absolute;
    inset: 12px;
    border-radius: 26px;
    border: 1px solid rgba(255,255,255,0.08);
    pointer-events: none;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 12px;
}
.hero p {
    color: #cbd5f5;
    line-height: 1.7;
}
.chip {
    background: rgba(8, 145, 178, 0.22);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.9rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 18px;
    margin-top: 18px;
}
.metric-card {
    background: var(--bg-card);
    padding: 18px;
    border-radius: 22px;
    border: 1px solid rgba(226, 232, 240, 0.06);
    backdrop-filter: blur(10px);
}
.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 4px 0 2px;
}
.metric-helper {
    font-size: 0.85rem;
    color: rgba(148, 163, 184, 0.9);
}
.section-card {
    background: rgba(2, 6, 23, 0.75);
    padding: 24px 28px;
    border-radius: 28px;
    border: 1px solid rgba(56,189,248,0.2);
    box-shadow: 0 20px 35px rgba(2, 6, 23, 0.6);
}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(130deg, #34d399, #22d3ee);
    color: #041016;
    border-radius: 999px;
    border: none;
    padding: 0.85rem 1.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.stButton>button:hover {
    filter: brightness(1.08);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
    padding-top: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(8, 145, 178, 0.18);
    padding: 12px 22px;
    border-radius: 18px;
    border: 1px solid transparent;
    color: #e0f2fe;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(100deg, rgba(14,165,233,0.4), rgba(236,72,153,0.15));
    border-color: rgba(14,165,233,0.6);
    color: #0f172a !important;
}
.stDataFrame {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(226, 232, 240, 0.08);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="chip">💡 Insight-ready ML dashboard</div>
        <h1>Salary Intelligence Canvas</h1>
        <p>Blend exploratory analytics with instant predictions. Compare real job postings,
           surface patterns in the dataset, and sculpt custom scenarios to see how the model
           reacts in real time.</p>
        <div class="chip">Built with Streamlit · Altair · scikit-learn</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if DATA_PATH is None:
    st.error(
        "❌ Could not find the cleaned salary CSV file "
        "(searched for `salary_data_cleaned.csv`).\n\n"
        "Make sure it is inside a folder in this project (for example `data/`)."
    )
    st.stop()

st.caption(
    "Dataset source: [Jobs Dataset from Glassdoor (Kaggle)]"
    "(https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor?resource=download)"
)

df = pd.read_csv(DATA_PATH)

ESSENTIAL_COLUMNS = [
    "Job Title",
    "Company Name",
    "Location",
    "Industry",
    "Sector",
    "Rating",
    "avg_salary",
    "min_salary",
    "max_salary",
    "Salary Estimate",
    "Job Description",
]

if MODEL_PATH is None:
    st.warning(
        "⚠️ Trained model file `salary_model.pkl` not found.\n"
        "You can still explore the dataset below, "
        "but predictions will be disabled."
    )
    model = None
    artifact = None
    feature_names = df.columns.tolist()
    target_col = None
else:
    st.success(
        f"✅ Loaded trained model from: `{MODEL_PATH.relative_to(BASE_DIR)}`"
    )
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_col = artifact.get("target_col", None)

# ---------- 2. Dataset overview ----------

with st.sidebar:
    st.header("Pilot instructions")
    st.markdown(
        "1. **Explore** high-level signals.\n"
        "2. **Inspect** real postings for ground truth.\n"
        "3. **Experiment** with a custom scenario."
    )
    st.markdown("---")
    st.caption("Active data source")
    st.code(f"{DATA_PATH.relative_to(BASE_DIR)}")
    if MODEL_PATH is not None:
        st.caption("Model artifact")
        st.code(f"{MODEL_PATH.relative_to(BASE_DIR)}")
    else:
        st.warning("Model not found — predictions disabled.")

def metric_card(label: str, value: str, helper: str) -> None:
    """Render a stylized metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float | int | None, decimals: int = 0) -> str:
    """Return a formatted salary string assuming value represents thousands of USD."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    return f"${value:,.{decimals}f}K"


def format_currency_delta(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.{decimals}f}K"


tabs = st.tabs(["Insight Board", "Existing Posting", "Custom Lab"])

with tabs[0]:
    st.subheader("Insight board")
    total_rows = f"{len(df):,}"
    total_cols = f"{len(df.columns):,}"
    unique_titles = (
        f"{df['Job Title'].nunique():,}"
        if "Job Title" in df.columns
        else "—"
    )
    unique_companies = (
        f"{df['Company Name'].nunique():,}"
        if "Company Name" in df.columns
        else "—"
    )
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    metric_card("Rows", total_rows, "Observations in dataset")
    metric_card("Columns", total_cols, "Features tracked")
    metric_card("Job titles", unique_titles, "Unique roles")
    metric_card("Companies", unique_companies, "Hiring orgs")
    st.markdown("</div>", unsafe_allow_html=True)

    salary_columns = [
        col for col in ["avg_salary", "min_salary", "max_salary"] if col in df.columns
    ]
    if salary_columns:
        st.markdown("#### Salary snapshot (values in thousands of USD)")
        salary_stats = []
        if "avg_salary" in df.columns:
            salary_stats.append(
                ("Average salary (mean)", format_currency(df["avg_salary"].mean(), 1))
            )
            salary_stats.append(
                ("Average salary (median)", format_currency(df["avg_salary"].median(), 1))
            )
        if "min_salary" in df.columns:
            salary_stats.append(
                ("Dataset min salary", format_currency(df["min_salary"].min(), 0))
            )
        if "max_salary" in df.columns:
            salary_stats.append(
                ("Dataset max salary", format_currency(df["max_salary"].max(), 0))
            )
        stat_cols = st.columns(max(1, len(salary_stats)))
        for col, (label, value) in zip(stat_cols, salary_stats):
            col.metric(label, value)

    with st.container():
        st.markdown("#### Smart preview")
        show_advanced = st.checkbox(
            "Show advanced engineer features",
            value=False,
            help="Toggle to reveal engineered or technical columns.",
        )
        preview_options = (
            [col for col in ESSENTIAL_COLUMNS if col in df.columns]
            if not show_advanced
            else list(df.columns)
        )
        preview_cols = st.multiselect(
            "Columns to display",
            options=preview_options,
            default=preview_options[: min(6, len(preview_options))],
            key="preview_cols",
        )
        rows_to_show = st.slider(
            "Rows to preview",
            min_value=5,
            max_value=min(50, len(df)),
            value=min(15, len(df)),
            step=5,
            key="preview_rows",
        )
        preview_df = df.head(rows_to_show)
        if preview_cols:
            preview_df = preview_df[preview_cols]
        st.dataframe(preview_df, use_container_width=True)

    viz_col1, viz_col2 = st.columns(2)

    if "Industry" in df.columns:
        with viz_col1:
            st.markdown("#### Top industries by posting volume")
            top_industries = (
                df["Industry"]
                .fillna("Unknown")
                .value_counts()
                .head(7)
                .reset_index()
            )
            top_industries.columns = ["Industry", "count"]
            if not top_industries.empty:
                top_industries["share_pct"] = (
                    top_industries["count"] / len(df) * 100.0
                )
                top_industries["share_label"] = top_industries["share_pct"].map(
                    lambda x: f"{x:.1f}%"
                )
                base_chart = alt.Chart(top_industries).encode(
                    x=alt.X(
                        "count:Q",
                        title="Number of postings",
                        axis=alt.Axis(format=","),
                    ),
                    y=alt.Y("Industry:N", sort="-x", title="Industry"),
                    tooltip=[
                        alt.Tooltip("Industry:N"),
                        alt.Tooltip("count:Q", title="Postings", format=","),
                        alt.Tooltip("share_label:N", title="Dataset share"),
                    ],
                )
                bars = base_chart.mark_bar(
                    cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#38bdf8"
                )
                labels = base_chart.mark_text(
                    dx=8,
                    align="left",
                    color="#f8fafc",
                    fontWeight=600,
                ).encode(text="share_label:N")
                st.altair_chart(bars + labels, use_container_width=True)
            else:
                st.info("The Industry column has no values to visualize yet.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_for_plot = None
    if target_col and target_col in numeric_cols:
        target_for_plot = target_col
    elif numeric_cols:
        target_for_plot = numeric_cols[0]

    if target_for_plot:
        with viz_col2:
            st.markdown(f"#### Distribution of `{target_for_plot}`")
            series = df[target_for_plot].dropna()
            is_currency_target = target_for_plot in {
                "avg_salary",
                "min_salary",
                "max_salary",
            }
            stats_cols = st.columns(3)
            if is_currency_target:
                stats_cols[0].metric("Mean", format_currency(series.mean(), 1))
                stats_cols[1].metric("Median", format_currency(series.median(), 1))
                stats_cols[2].metric("Std dev", format_currency(series.std(), 1))
            else:
                stats_cols[0].metric("Mean", f"{series.mean():.2f}")
                stats_cols[1].metric("Median", f"{series.median():.2f}")
                stats_cols[2].metric("Std dev", f"{series.std():.2f}")
            hist_chart = (
                alt.Chart(df)
                .mark_area(
                    line={"color": "#34d399", "strokeWidth": 2},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            {"color": "#22d3ee55", "offset": 0},
                            {"color": "#34d39911", "offset": 1},
                        ],
                    ),
                    opacity=0.9,
                )
                .encode(
                    x=alt.X(
                        f"{target_for_plot}:Q",
                        bin=alt.Bin(maxbins=25),
                        axis=alt.Axis(
                            format="$,.0f" if is_currency_target else "",
                            title="Salary (thousand USD)"
                            if is_currency_target
                            else f"{target_for_plot} value range",
                        ),
                    ),
                    y=alt.Y("count()", title="Number of postings"),
                    tooltip=[
                        alt.Tooltip(
                            f"{target_for_plot}:Q",
                            title="Salary ($K)" if is_currency_target else "Value",
                            format="$,.0f" if is_currency_target else None,
                        ),
                        alt.Tooltip("count():Q", title="Postings"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(hist_chart, use_container_width=True)

    if {"Rating", target_for_plot}.issubset(set(df.columns)):
        st.markdown("#### Rating vs. target relationship")
        scatter_df = df.dropna(subset=["Rating", target_for_plot]).copy()
        scatter_df = scatter_df.head(1000)  # control rendering cost
        is_currency_target = target_for_plot in {
            "avg_salary",
            "min_salary",
            "max_salary",
        }
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=110, opacity=0.7)
            .encode(
                x=alt.X("Rating:Q", scale=alt.Scale(domain=[0, 5]), title="Company rating"),
                y=alt.Y(
                    f"{target_for_plot}:Q",
                    title="Salary (thousand USD)" if is_currency_target else target_for_plot,
                    axis=alt.Axis(format="$,.0f" if is_currency_target else ""),
                ),
                color=alt.Color(
                    "Rating:Q",
                    scale=alt.Scale(scheme="tealblues"),
                    legend=alt.Legend(title="Rating score"),
                ),
                tooltip=[
                    alt.Tooltip("Rating:Q"),
                    alt.Tooltip(
                        f"{target_for_plot}:Q",
                        title="Salary ($K)" if is_currency_target else target_for_plot,
                        format="$,.0f" if is_currency_target else None,
                    ),
                    alt.Tooltip("Company Name:N", title="Company"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(scatter, use_container_width=True)

    if {"Industry", "avg_salary"}.issubset(df.columns):
        st.markdown("#### Industries by average salary")
        salary_by_industry = (
            df.groupby("Industry", dropna=False)["avg_salary"]
            .mean()
            .dropna()
            .reset_index()
            .sort_values("avg_salary", ascending=False)
            .head(7)
        )
        if not salary_by_industry.empty:
            salary_by_industry["salary_fmt"] = salary_by_industry["avg_salary"].map(
                lambda x: format_currency(x, 1)
            )
            salary_chart = (
                alt.Chart(salary_by_industry)
                .mark_bar(cornerRadius=10, color="#f472b6")
                .encode(
                    x=alt.X(
                        "avg_salary:Q",
                        title="Average salary (thousand USD)",
                        axis=alt.Axis(format="$,.0f"),
                    ),
                    y=alt.Y("Industry:N", sort="-x"),
                    tooltip=[
                        alt.Tooltip("Industry:N"),
                        alt.Tooltip("salary_fmt:N", title="Average salary"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(salary_chart, use_container_width=True)
        else:
            st.info("Insufficient salary data per industry to visualize.")

# ---------- 3. Prediction from an existing job posting ----------

if "selected_existing_idx" not in st.session_state:
    st.session_state["selected_existing_idx"] = 0

with tabs[1]:
    st.subheader("Make a prediction from an existing job posting")
    st.caption(
        "Choose a row from the dataset to inspect the raw features and compare "
        "the predicted vs. actual target (when available)."
    )
    col_left, col_right = st.columns([1, 2])

    with col_left:
        if "Job Title" in df.columns:
            search_term = st.text_input(
                "Quick search (job title contains)",
                "",
                placeholder="e.g., data scientist",
                key="existing_search",
            )
            if search_term:
                mask = df["Job Title"].astype(str).str.contains(
                    search_term, case=False, na=False
                )
                if mask.any():
                    jump_idx = int(df[mask].index[0])
                    st.session_state["selected_existing_idx"] = jump_idx
                    st.info(f"Jumped to row {jump_idx}")
                else:
                    st.warning("No matching titles found.")

        if st.button("🎲 Surprise me", use_container_width=True):
            st.session_state["selected_existing_idx"] = random.randint(0, len(df) - 1)

        st.slider(
            "Pick a row index",
            0,
            len(df) - 1,
            st.session_state["selected_existing_idx"],
            key="selected_existing_idx",
            help="Move through real postings.",
        )
        st.caption("Use the search box, slider, or surprise button to browse postings.")

    selected_idx = st.session_state["selected_existing_idx"]
    row = df.iloc[[selected_idx]]  # keep as DataFrame

    with col_right:
        st.write(f"Selected row: **{selected_idx}**")
        highlight_cols = []
        display_info = []
        if "Job Title" in row.columns:
            display_info.append(("Job title", row["Job Title"].iloc[0]))
            highlight_cols.append("Job Title")
        if "Company Name" in row.columns:
            display_info.append(("Company", row["Company Name"].iloc[0]))
            highlight_cols.append("Company Name")
        if "Location" in row.columns:
            display_info.append(("Location", row["Location"].iloc[0]))
            highlight_cols.append("Location")
        if "Rating" in row.columns:
            display_info.append(("Rating", f"{row['Rating'].iloc[0]:.2f}"))
            highlight_cols.append("Rating")
        if display_info:
            info_cols = st.columns(len(display_info))
            for col, (label, value) in zip(info_cols, display_info):
                col.markdown(f"**{label}**")
                col.markdown(f"<div class='chip'>{value}</div>", unsafe_allow_html=True)

        salary_display = []
        for label, column in [
            ("Min salary", "min_salary"),
            ("Avg salary", "avg_salary"),
            ("Max salary", "max_salary"),
        ]:
            if column in row.columns:
                salary_display.append((label, format_currency(row[column].iloc[0], 1)))
                highlight_cols.append(column)
        if salary_display:
            st.markdown("**Compensation window**")
            salary_cols = st.columns(len(salary_display))
            for col, (label, value) in zip(salary_cols, salary_display):
                col.metric(label, value)

        essential_available = [
            col for col in ESSENTIAL_COLUMNS if col in row.columns and col not in highlight_cols
        ]
        selectable_columns = list(dict.fromkeys(highlight_cols + essential_available))
        display_subset = st.multiselect(
            "Columns to show in the grid",
            options=selectable_columns,
            default=selectable_columns,
            help="Advanced engineered features stay hidden to keep things readable.",
        )
        if st.checkbox("Show all raw columns", value=False):
            display_subset = list(row.columns)
        st.dataframe(
            row[display_subset] if display_subset else row,
            use_container_width=True,
        )
        st.download_button(
            "Download this row (CSV)",
            row.to_csv(index=False).encode("utf-8"),
            file_name=f"job_row_{selected_idx}.csv",
            mime="text/csv",
        )
        with st.expander("Full feature vector"):
            st.dataframe(row, use_container_width=True)

    if model is not None:
        X_row = row[feature_names]
        pred = model.predict(X_row)[0]

        if target_col is not None and target_col in row.columns:
            real = row[target_col].iloc[0]
        else:
            real = None

        st.markdown("### Prediction")
        if real is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Predicted value",
                format_currency(pred, 1),
                delta=format_currency_delta(pred - real, 1) + " vs actual",
            )
            c2.metric("Actual value", format_currency(real, 1))
            c3.metric(
                "Absolute error",
                format_currency(abs(pred - real), 1),
                delta="Lower is better",
            )
        else:
            st.metric("Predicted value", format_currency(pred, 1))
            st.caption("Real target value not available in the saved model.")
    else:
        st.info("Train and save a model as `salary_model.pkl` to enable predictions.")

# ---------- 4. Custom job prediction form ----------

with tabs[2]:
    st.subheader("Custom job prediction")

    st.caption(
        "Start from a real job in the dataset, edit some fields (title, company, "
        "location, rating, etc.), and let the model predict the target value "
        "(for example average salary)."
    )

    if model is None:
        st.info("Custom prediction is disabled because no model is loaded.")
    else:
        # Choose a base row to copy defaults from
        base_index = st.number_input(
            "Base row index (template from dataset)",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
        )
        base_row = df.iloc[base_index].copy()

        with st.form("custom_job_form"):
            st.write("Edit the fields you want to change:")

            # Config of human-friendly fields (only used if column exists)
            widgets_config = [
                ("Job Title", "text", {}),
                ("Company Name", "text", {}),
                ("Location", "text", {}),
                ("Rating", "float", {"min": 0.0, "max": 5.0, "step": 0.1}),
                ("Size", "text", {}),
                ("Industry", "text", {}),
                ("Sector", "text", {}),
                ("Founded", "int", {"min": 1800, "max": 2100, "step": 1}),
            ]

            custom_row = base_row.copy()

            for col, kind, params in widgets_config:
                if col not in df.columns:
                    continue  # skip if this column doesn't exist

                default_val = base_row[col]

                if kind == "text":
                    val = st.text_input(col, value=str(default_val))
                elif kind == "float":
                    try:
                        default_num = float(default_val)
                    except Exception:
                        default_num = 0.0
                    val = st.number_input(
                        col,
                        min_value=params.get("min", 0.0),
                        max_value=params.get("max", 1000.0),
                        step=params.get("step", 0.1),
                        value=default_num,
                    )
                elif kind == "int":
                    try:
                        default_int = int(default_val)
                    except Exception:
                        default_int = 2000
                    val = st.number_input(
                        col,
                        min_value=params.get("min", 1800),
                        max_value=params.get("max", 2100),
                        step=params.get("step", 1),
                        value=default_int,
                    )
                else:
                    val = default_val

                custom_row[col] = val

            submitted = st.form_submit_button("⚡ Predict this scenario")

        if submitted:
            # Start from base_row so all technical features stay valid
            row_for_model = base_row.copy()
            for col in custom_row.index:
                row_for_model[col] = custom_row[col]

            X_custom = pd.DataFrame([row_for_model])[feature_names]
            pred_custom = model.predict(X_custom)[0]

            st.markdown("### Prediction for custom job")
            st.success(f"Predicted {target_col or 'value'}: **{pred_custom:.2f}**")

            st.write("Custom job data used for prediction:")
            st.dataframe(pd.DataFrame([row_for_model]), use_container_width=True)

st.markdown("---")
st.caption(
    "This app is part of the **Software Engineering for AI Projects** class. "
    "It loads a real dataset, trains a model, and provides a web interface "
    "to explore the data and make predictions."
)
