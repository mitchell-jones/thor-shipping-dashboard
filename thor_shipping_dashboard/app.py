from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.dates as mdates
from thor_shipping_dashboard.scrape import scrape_ayn_dashboard


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load shipping progress data from JSON and normalize types.

    Expected columns: ``date``, ``variant``, ``color``, ``start_prefix``,
    ``end_prefix``. Dates are parsed as naive timestamps in YYYY-MM-DD.
    """
    data_path = get_project_root() / "data" / "shipping_progress.json"
    df = pd.read_json(data_path)

    # Dates are authored as YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=False)

    # Ensure expected columns exist in the current schema
    required_cols = {"variant", "color", "start_prefix", "end_prefix"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {sorted(missing)}")

    df["start_prefix"] = df["start_prefix"].astype(int)
    df["end_prefix"] = df["end_prefix"].astype(int)
    return df




def get_selection_options(df: pd.DataFrame) -> Tuple[list[str], dict[str, Tuple[str, str]]]:
    """Build user-facing model options and a mapping to (variant, color).

    Returns a sorted list of display strings in the form
    ``"{color} | {variant}"`` and a mapping from that string to a
    ``(variant, color)`` tuple used for filtering.
    """
    # Build only valid combinations from unique pairs; use a delimiter safe for spaces
    unique_pairs = (
        df[["color", "variant"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["color", "variant"])  # stable ordering
    )
    display_values: list[str] = []
    mapping: dict[str, Tuple[str, str]] = {}
    for _, row in unique_pairs.iterrows():
        display = f"{row['color']} | {row['variant']}"
        display_values.append(display)
        mapping[display] = (row["variant"], row["color"])  # (variant, color)
    return display_values, mapping


def filter_data(df: pd.DataFrame, variant: str, color: str) -> pd.DataFrame:
    """Filter the dataset to the selected ``variant`` and ``color``.

    Ensures one row per date (keeping the last entry when duplicates exist)
    and returns the result ordered by date.
    """
    sdf = df[(df["variant"].astype(str) == variant) & (df["color"].astype(str) == color)].copy()
    sdf = sdf.sort_values("date")
    # Ensure one row per date (keep the last/latest if duplicates exist)
    sdf = sdf.groupby("date", as_index=False).last()
    return sdf


def plot_progress(sdf: pd.DataFrame) -> plt.Figure:
    """Create the base shipping progress figure for the filtered data."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sdf["date"], sdf["end_prefix"], marker="o", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Latest Order Number Prefix")
    ax.set_title("Shipping Progress by Day")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.autofmt_xdate()
    return fig


def add_trend_and_estimate(ax: plt.Axes, sdf: pd.DataFrame, target_prefix: int | None) -> tuple[pd.Timestamp | None, str | None, float | None, int | None]:
    """Overlay a linear trend and compute estimates.

    - Fits a simple linear regression of ``end_prefix`` over time.
    - Draws a dashed trend line from the first observed date through either
      the last observed date or the user's estimated ship date (if later).
    - If ``target_prefix`` is provided, computes the estimated ship date,
      the shipment rate (units/day), and remaining shipments to the target.

    Returns a tuple of ``(estimated_date, warning, units_per_day, remaining)``.
    """
    if len(sdf) < 2:
        return None, "Only one data point available; unable to fit a trend line.", None, None

    # Use continuous matplotlib date numbers (days as float) to avoid stepwise trend
    x = sdf["date"].map(mdates.date2num).to_numpy(dtype=float)
    y = sdf["end_prefix"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    est_date: pd.Timestamp | None = None
    warn: str | None = None

    # Determine end of trend line
    dmin, dmax = sdf["date"].min(), sdf["date"].max()
    trend_end = dmax

    if target_prefix is not None:
        if slope <= 0:
            warn = "Trend slope is non-positive; cannot estimate shipping date."
        else:
            t_est = (float(target_prefix) - intercept) / slope  # matplotlib date number
            est_dt = mdates.num2date(t_est)
            est_date = pd.Timestamp(est_dt).tz_localize(None)
            # Extend trend to the estimate point if it's after last observed date
            if est_date > trend_end:
                trend_end = est_date
            if est_date < dmin or est_date > dmax + pd.Timedelta(days=365):
                warn = "Estimated date falls outside the observed range; treat with caution."

    # Trend line from first day through to trend_end
    xs = np.linspace(mdates.date2num(dmin), mdates.date2num(trend_end), 400)
    ys = slope * xs + intercept
    xs_dates = [mdates.num2date(xi) for xi in xs]
    ax.plot(xs_dates, ys, linestyle="--", color="tab:orange", label="Linear trend")

    # Removed on-plot annotation; we'll display units/day beneath the chart

    if est_date is not None and target_prefix is not None:
        ax.scatter([est_date], [target_prefix], color="tab:red", zorder=5, label="Your est. ship date")
        ax.axvline(est_date, color="tab:red", linestyle=":", alpha=0.6)

    # Ensure legend is visible if we added anything
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    remaining: int | None = None
    if target_prefix is not None:
        latest = int(sdf["end_prefix"].iloc[-1])
        # Convert prefix delta to units (each prefix step represents 1000 units)
        remaining = max(0, (int(target_prefix) - latest) * 1000)

    # Convert slope (prefix/day) to units/day
    units_per_day = float(slope) * 1000.0
    return est_date, warn, units_per_day, remaining


def main() -> None:
    """Streamlit entrypoint to render selectors, chart, and estimates."""
    st.set_page_config(page_title="AYN Thor Shipping Dashboard", layout="centered")
    st.title("AYN Thor Shipping Dashboard")
    st.caption("Visualize shipping progress by Variant and Colorway.")

    use_live = st.toggle("Use live data from AYN dashboard", value=False, help="Scrape the latest from ayntec.com; falls back to bundled JSON if it fails.")
    if use_live:
        try:
            df = scrape_ayn_dashboard()
            st.caption("Loaded live data from AYN dashboard.")
        except Exception as exc:
            st.warning(f"Live load failed: {exc}. Using bundled JSON instead.")
            df = load_data()
    else:
        df = load_data()
    combos, combo_to_parts = get_selection_options(df)

    selected_combo = st.selectbox("Model", combos, index=0 if combos else None)

    if not selected_combo:
        st.info("No options available. Please check the data file.")
        return

    variant, color = combo_to_parts[selected_combo]
    sdf = filter_data(df, variant, color)
    if sdf.empty:
        st.warning("No shipping data for the selected configuration.")
        return

    # Primary chart
    fig = plot_progress(sdf)

    # User input for personal order number prefix
    st.subheader("Estimate Your Shipping Date")
    target_prefix = st.number_input(
        "Enter your order number prefix (first three digits, e.g., 940)",
        min_value=100,
        max_value=12000,
        step=1,
        value=None,
        placeholder="e.g., 940",
    )

    # Overlay trend and estimate on the existing figure (not a new one)
    ax = fig.axes[0]
    est_date, warn, units_per_day, remaining = add_trend_and_estimate(ax, sdf, int(target_prefix) if target_prefix is not None else None)
    st.pyplot(fig, clear_figure=True)
    # Summary text under the chart
    if units_per_day is not None:
        st.markdown(f"**Estimated rate:** {units_per_day:,.0f} units/day")
    if target_prefix is not None and remaining is not None:
        st.markdown(f"**Shipments to go:** {remaining:,}")
    if est_date is not None:
        st.markdown(f"**Estimated shipping date:** {est_date.date()}")
    if warn:
        st.info(warn)

    with st.expander("Show data:", expanded=False):
        st.dataframe(sdf[["date", "start_prefix", "end_prefix"]], use_container_width=True, hide_index=True)

    # Diagnose selections without graphing data
    if len(sdf) == 0:
        st.info("No data points for this selection. This model/color may not have reported shipments yet.")
    elif len(sdf) == 1:
        st.info("Only one data point exists for this selection; trendline and estimate are disabled.")


if __name__ == "__main__":
    main()



