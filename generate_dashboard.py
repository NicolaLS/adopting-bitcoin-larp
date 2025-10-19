#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
import textwrap
from typing import Dict, List, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
OVERVIEW_PATH = ROOT / "conferences_overview.csv"
DETAIL_PATH = ROOT / "absv2024_stats_detail.csv"
OUTPUT_DIR = ROOT / "build"
OUTPUT_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "res"

BRAND_FONT = "Poppins, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
BRAND_BACKGROUND = "#05051f"
BRAND_SURFACE = "#0b0f2f"
BRAND_PANEL = "rgba(22, 23, 58, 0.85)"
BRAND_BORDER = "#1d2256"
BRAND_TEXT = "#f1f3ff"
BRAND_MUTED = "#9ca3e1"
BRAND_ACCENT = "#EEDB5F"
BRAND_MAGENTA = "#FF58A7"
BRAND_ELECTRIC = "#0400FF"
BRAND_COLORWAY = [
    BRAND_ACCENT,
    BRAND_MAGENTA,
    BRAND_ELECTRIC,
    "#5a5df0",
    "#23c0f5",
    "#fba82b",
]
BRAND_GRID = "#1c214d"
ENABLE_PAGE_GRADIENT = False
PAGE_GRADIENT_CSS = (
    "radial-gradient(circle at 20% 20%, rgba(255, 88, 167, 0.28), transparent 55%),"
    " radial-gradient(circle at 80% 0%, rgba(4, 0, 255, 0.35), transparent 45%),"
    " var(--page-bg)"
)

LIGHTNING_STORY_HTML = """
<div class="story-block">
    <h3>Every corner buzzed with Lightning.</h3>
    <p>We encourage every attendee to tap lightning from the moment they set foot in El Salvador. The conference venue is packed with food and beverage vendors running on sats, letting lunch and coffee settle instantly via bitcoin.</p>
    <p>When the daytime programming wraps, afterparties keep the tabs open in sats so cocktails and late-night snacks ride the same rails.</p>
    <p>Art lovers made history: more than 1 BTC worth of original pieces left the gallery floor, proving that culture trades hands faster when artists and collectors share a lightning wallet.</p>
    <p>ChainDuel battles triggered a stream of lightning wagers, the poker tables ran on the same rails, and there was even a tattoo artist inking designs for sats. It’s bitcoin country here, and the network never sleeps.</p>
</div>
"""


TRANSACTION_STORY_HTML = """
<div class="story-block">
    <h3>Transactions are the mission.</h3>
    <p>We design every touchpoint to reward lightning usage, and it shows: 2,969 payments lit up Adopting Bitcoin 2024 because spending sats is the default here.</p>
    <p>Food and merch partners racked up 1,761 checkouts, afterparties added 1,027 joyful taps, and ChainDuel matches alone fired off 105 lightning transactions&mdash;proof that incentives and playful prompts get wallets buzzing.</p>
    <p>We’re proud to stand shoulder to shoulder with fellow conferences: our 2.47 tx per attendee outruns BTC Prague (0.73) and even the massive Bitcoin 2024 (0.04), while the data nudges us to learn from high-intent gatherings like Bitcoin Freedom Festival’s 14.48 tx-per-guest playbook so we can keep raising the bar next year.</p>
</div>
""".strip()


KPI_CARD_STYLE = """
.kpi-card {
    position: relative;
    width: 100%;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 16px;
    padding: 22px 26px 20px;
    border-radius: 22px;
    background: rgba(11, 14, 42, 0.82);
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 22px 45px rgba(4, 5, 22, 0.45);
    overflow: hidden;
}
.kpi-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(238, 219, 95, 0.35), rgba(255, 88, 167, 0.08) 45%, transparent 85%);
    pointer-events: none;
}
.kpi-card > * {
    position: relative;
    z-index: 1;
}
.kpi-label {
    text-transform: uppercase;
    letter-spacing: 0.24em;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(238, 219, 95, 0.9);
}
.kpi-value {
    font-size: 2.7rem;
    font-weight: 600;
    line-height: 1.05;
    color: #fdfdff;
    text-shadow: 0 6px 22px rgba(4, 0, 255, 0.22);
}
.kpi-footnote {
    font-size: 0.95rem;
    color: rgba(156, 163, 225, 0.92);
}
""".strip()

BRAND_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=BRAND_FONT, color=BRAND_TEXT),
        paper_bgcolor=BRAND_SURFACE,
        plot_bgcolor=BRAND_SURFACE,
        colorway=BRAND_COLORWAY,
        title=dict(font=dict(size=20, color=BRAND_TEXT)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=BRAND_MUTED),
            borderwidth=0,
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=BRAND_GRID,
            zerolinecolor=BRAND_GRID,
            linecolor=BRAND_GRID,
            tickfont=dict(color=BRAND_MUTED),
            title=dict(font=dict(color=BRAND_MUTED)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=BRAND_GRID,
            zerolinecolor=BRAND_GRID,
            linecolor=BRAND_GRID,
            tickfont=dict(color=BRAND_MUTED),
            title=dict(font=dict(color=BRAND_MUTED)),
        ),
    )
)

px.defaults.template = BRAND_TEMPLATE
px.defaults.color_discrete_sequence = BRAND_COLORWAY


def slugify(label: str) -> str:
    """Create filesystem-friendly slugs for filenames."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace("\"", "", regex=False)
    cleaned = cleaned.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(cleaned, errors="coerce")


def wrap_title(text: str, width: int = 28) -> str:
    if not text:
        return text
    lines = textwrap.wrap(text, width, break_long_words=False)
    return "<br>".join(lines)


def format_compact(value: float | int, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    for suffix, threshold in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if magnitude >= threshold:
            scaled = magnitude / threshold
            return f"{sign * scaled:.{decimals}f}{suffix}"
    if magnitude >= 1:
        return f"{sign * magnitude:.0f}"
    return f"{sign * magnitude:.2f}"


def format_metric(value: float | int | None, format_spec) -> str:
    if value is None or pd.isna(value):
        return "—"
    if callable(format_spec):
        return format_spec(value)
    return format(value, format_spec)


def render_kpi_markup(kpi: Dict[str, object], *, show_label: bool) -> str:
    label = f'<span class="kpi-label">{kpi["title"]}</span>' if show_label else ""
    formatted_value = format_metric(kpi["value"], kpi["format"])
    suffix = kpi.get("suffix", "")
    value_color = kpi.get("value_color")
    color_attr = f' style="color:{value_color};"' if value_color else ""
    footnote = kpi.get("footnote")
    footnote_html = f'<span class="kpi-footnote">{footnote}</span>' if footnote else ""
    return (
        f'<div class="kpi-card">{label}'
        f'<span class="kpi-value"{color_attr}>{formatted_value}{suffix}</span>'
        f"{footnote_html}</div>"
    )


def write_kpi_component(kpi: Dict[str, object]) -> str:
    kpi_dir = OUTPUT_DIR / "kpi"
    kpi_dir.mkdir(exist_ok=True)
    slug = kpi.get("slug") or slugify(kpi["title"])
    component_html = render_kpi_markup(kpi, show_label=False)
    full_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>{kpi['title']} – KPI</title>
    <style>
        :root {{
            color-scheme: dark;
            font-family: {BRAND_FONT};
            --accent: {BRAND_ACCENT};
            --muted: {BRAND_MUTED};
        }}
        body {{
            margin: 0;
            background: {BRAND_BACKGROUND};
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px;
            color: {BRAND_TEXT};
        }}
        {KPI_CARD_STYLE}
    </style>
</head>
<body>
    {render_kpi_markup(kpi, show_label=True)}
</body>
</html>
"""
    (kpi_dir / f"{slug}.html").write_text(full_html, encoding="utf-8")
    return component_html


def load_overview() -> pd.DataFrame:
    df = pd.read_csv(OVERVIEW_PATH, skip_blank_lines=True, engine="python")
    df = df[df["Date"].notna()]
    df = df[~df["Date"].astype(str).str.startswith("This includes")]
    numeric_cols = [
        "Attendance",
        "Transactions",
        "txs per Attendee",
        "Volume (BTC)",
        "Volume (SAT)",
        "Avg tx size (sats)",
        "Revenue per attendee (sats)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_numeric(df[col])
    df = df.rename(
        columns={
            "Event Name": "Event",
            "City": "Location",
            "txs per Attendee": "TxPerAttendee",
            "Volume (BTC)": "VolumeBTC",
            "Volume (SAT)": "VolumeSAT",
            "Avg tx size (sats)": "AvgTxSizeSats",
            "Revenue per attendee (sats)": "RevenuePerAttendeeSats",
            "Conference Size": "ConferenceSize",
        }
    )
    df["Latest"] = df["Latest"].astype(str).str.upper().eq("TRUE")
    df["DateParsed"] = pd.to_datetime(df["Date"], format="%B %Y")
    df["Year"] = df["DateParsed"].dt.year
    df["MonthLabel"] = df["DateParsed"].dt.strftime("%b %Y")
    df["EventLabel"] = df["Event"] + " — " + df["Location"]
    return df


def load_details() -> pd.DataFrame:
    df = pd.read_csv(DETAIL_PATH)
    df = df.rename(
        columns={
            "Volume (SAT)": "VolumeSAT",
            "Avg. Tx. Size": "AvgTxSize",
        }
    )
    df["Transactions"] = parse_numeric(df["Transactions"])
    df["VolumeSAT"] = parse_numeric(df["VolumeSAT"])
    df["AvgTxSize"] = parse_numeric(df["AvgTxSize"])
    df["Venue"] = df["Venue"].astype(str).str.upper().eq("TRUE")
    category_map = {
        "Vendors": "Vendors",
        "Rebel Beer": "Vendors",
        "ChainDuel at AB24": "ChainDuel",
        "ChainDuel at ABAP": "ChainDuel",
    }
    df["CategoryGroup"] = df["Category"].map(category_map).fillna(df["Category"])
    df["VolumeBTC"] = df["VolumeSAT"] / 1e8
    return df


def create_ab24_kpis(ab24_row: pd.Series) -> List[Dict[str, object]]:
    compact = lambda v: format_compact(float(v), decimals=1)
    return [
        {
            "title": "Attendees",
            "value": ab24_row["Attendance"],
            "format": compact,
            "suffix": "",
            "footnote": "Lightning-ready participants",
            "slug": "attendees",
        },
        {
            "title": "Transactions",
            "value": ab24_row["Transactions"],
            "format": compact,
            "suffix": "",
            "footnote": "Lightning transactions processed",
            "slug": "transactions",
        },
        {
            "title": "Volume (SAT)",
            "value": ab24_row["VolumeSAT"],
            "format": compact,
            "suffix": "",
            "footnote": "Total lightning spending",
            "slug": "volume-sat",
        },
        {
            "title": "Tx per Attendee",
            "value": ab24_row["TxPerAttendee"],
            "format": lambda v: f"{float(v):.2f}",
            "suffix": "x",
            "footnote": "Average lightning engagement",
            "slug": "tx-per-attendee",
        },
    ]


def create_ab24_sankey(detail_df: pd.DataFrame) -> go.Figure:
    detail_df = detail_df.copy()
    sankey_label_map = {
        "Rebel Beer": "Vendors",
        "Vendors": "Vendors",
        "ChainDuel at AB24": "ChainDuel",
        "ChainDuel at ABAP": "ChainDuel",
    }
    detail_df["SankeyCategory"] = detail_df["Category"].replace(sankey_label_map)

    category_totals = (
        detail_df.groupby("SankeyCategory")["VolumeBTC"].sum().sort_values(ascending=False)
    )
    category_order = category_totals.index.tolist()
    sankey_nodes = ["Adopting Bitcoin 2024"] + category_order
    node_index: Dict[str, int] = {name: idx for idx, name in enumerate(sankey_nodes)}

    flows: List[tuple[str, str, float]] = []
    for category, volume in category_totals.items():
        flows.append(("Adopting Bitcoin 2024", category, float(volume)))

    source = [node_index[src] for src, _, _ in flows]
    target = [node_index[tgt] for _, tgt, _ in flows]
    value = [max(v, 0.0) for _, _, v in flows]

    palette = BRAND_COLORWAY
    gradient_start = BRAND_MAGENTA
    gradient_end = BRAND_ELECTRIC
    right_colors: List[str] = []
    num_categories = len(category_order)
    for idx in range(num_categories):
        if num_categories <= 1:
            t = 0.0
        else:
            t = idx / (num_categories - 1)
        start_r = int(gradient_start[1:3], 16)
        start_g = int(gradient_start[3:5], 16)
        start_b = int(gradient_start[5:7], 16)
        end_r = int(gradient_end[1:3], 16)
        end_g = int(gradient_end[3:5], 16)
        end_b = int(gradient_end[5:7], 16)
        r = round(start_r + (end_r - start_r) * t)
        g = round(start_g + (end_g - start_g) * t)
        b = round(start_b + (end_b - start_b) * t)
        right_colors.append(f"#{r:02X}{g:02X}{b:02X}")

    node_colors = [BRAND_ACCENT] + right_colors

    root_x, root_y = 0.02, 0.5
    category_x = 0.97
    num_categories = len(category_order)
    if num_categories > 1:
        span = 0.88
        start = 0.06
        step = span / (num_categories - 1)
        category_y_positions = [start + step * idx for idx in range(num_categories)]
    elif num_categories == 1:
        category_y_positions = [0.5]
    else:
        category_y_positions = []

    node_x_positions = [root_x] + [category_x] * num_categories
    node_y_positions = [root_y] + category_y_positions

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=20,
                label=[""] + category_order,
                color=node_colors,
                line=dict(color=BRAND_BACKGROUND, width=1),
                x=node_x_positions,
                y=node_y_positions,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[BRAND_PANEL] * len(value),
                hovertemplate="%{source.label} → %{target.label}<br>%{value:.4f} BTC<extra></extra>",
            ),
        )
    )
    total_volume_btc = detail_df["VolumeBTC"].sum()
    fig.update_layout(
        title="",
        template=BRAND_TEMPLATE,
        height=620,
        margin=dict(t=48, l=16, r=16, b=36),
        paper_bgcolor=BRAND_PANEL,
    )
    return fig


def prepare_comparison_df(overview_df: pd.DataFrame) -> pd.DataFrame:
    latest = overview_df[overview_df["Latest"].fillna(False)].copy()
    latest = latest[latest["Year"] >= 2023].copy()
    latest = latest[~((latest["Event"] == "Adopting Bitcoin") & (latest["Year"] <= 2022))].copy()
    mask_abss_2024 = (
        (latest["Event"].str.contains("Adopting Bitcoin", case=False, na=False))
        & (latest["Location"].str.contains("San Salvador", na=False))
        & (latest["Year"] == 2024)
    )
    latest["IsABSS2024"] = mask_abss_2024
    return latest


def create_transactions_bar_fig(df: pd.DataFrame, *, show_labels: bool = True) -> go.Figure:
    plot_df = df.sort_values("Transactions", ascending=True).copy()
    plot_df["Color"] = plot_df["IsABSS2024"].map({True: BRAND_ACCENT, False: BRAND_MAGENTA})
    plot_df["BarLabel"] = plot_df["Event"]

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Transactions"]],
                y=[row["BarLabel"]],
                orientation="h",
                marker_color=row["Color"],
                text=[f"{row['Transactions']:,}"],
                textposition="outside",
                hovertemplate=(
                    f"{row['EventLabel']}<br>Transactions: {row['Transactions']:,}" + "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_traces(textfont=dict(color=BRAND_TEXT), cliponaxis=False)
    left_margin = 80 if show_labels else 40
    yaxis_config = dict(automargin=True, ticklabelstandoff=12)
    if not show_labels:
        yaxis_config.update(showticklabels=False)

    fig.update_layout(
        title="",
        template=BRAND_TEMPLATE,
        height=540,
        margin=dict(t=100, l=left_margin, r=32, b=60),
        paper_bgcolor=BRAND_PANEL,
        xaxis_title="Transactions",
        showlegend=False,
        yaxis=yaxis_config,
    )
    return fig


def create_peer_scatter_fig(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["Group"] = plot_df["IsABSS2024"].map({True: "Adopting Bitcoin 2024", False: "Peer conference"})
    fig = px.scatter(
        plot_df,
        x="Attendance",
        y="Transactions",
        size="VolumeBTC",
        color="Group",
        hover_data={
            "EventLabel": True,
            "Attendance": ":,.0f",
            "Transactions": ":,.0f",
            "TxPerAttendee": ":,.2f",
            "VolumeBTC": ":,.2f",
        },
        size_max=55,
        color_discrete_map={
            "Adopting Bitcoin 2024": BRAND_ACCENT,
            "Peer conference": BRAND_MAGENTA,
        },
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=BRAND_BACKGROUND)))
    fig.update_layout(
        title="",
        template=BRAND_TEMPLATE,
        height=540,
        margin=dict(t=100, l=80, r=60, b=80),
        paper_bgcolor=BRAND_PANEL,
        showlegend=False,
    )
    return fig


def create_on_off_site_split(detail_df: pd.DataFrame) -> go.Figure:
    summary = detail_df.groupby("Venue", as_index=False)[["Transactions", "VolumeBTC"]].sum()
    summary["Location"] = summary["Venue"].map({True: "On-site", False: "Off-site"})

    frames = []
    formatters = {
        "Transactions": lambda v: f"{v:,.0f} tx",
        "VolumeBTC": lambda v: f"{v:,.2f} BTC",
    }
    label_map = {"Transactions": "Transactions", "VolumeBTC": "Volume (BTC)"}
    for metric in ("Transactions", "VolumeBTC"):
        total = summary[metric].sum()
        share = summary[metric] / total * 100 if total else 0
        frame = pd.DataFrame(
            {
                "Metric": metric,
                "MetricLabel": label_map[metric],
                "Location": summary["Location"],
                "Share": share,
                "AbsoluteLabel": summary[metric].apply(formatters[metric]),
            }
        )
        frames.append(frame)

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["MetricLabel"] = pd.Categorical(
        plot_df["MetricLabel"], categories=["Transactions", "Volume (BTC)"], ordered=True
    )
    plot_df = plot_df.sort_values(["MetricLabel", "Location"], ascending=[True, False])

    fig = px.bar(
        plot_df,
        x="MetricLabel",
        y="Share",
        color="Location",
        text="Share",
        barmode="stack",
        color_discrete_map={"On-site": BRAND_ACCENT, "Off-site": BRAND_MAGENTA},
    )
    fig.update_traces(text=None)
    for trace in fig.data:
        mask = plot_df["Location"] == trace.name
        ordered = plot_df.loc[mask].sort_values("MetricLabel")
        trace.customdata = ordered["AbsoluteLabel"].to_numpy()
        trace.hovertemplate = "%{x}<br>%{customdata}<br>%{y:.1f}% share<extra></extra>"

    fig.update_layout(
        title="",
        template=BRAND_TEMPLATE,
        height=430,
        margin=dict(t=100, l=60, r=40, b=70),
        paper_bgcolor=BRAND_PANEL,
        font=dict(color=BRAND_TEXT),
        xaxis_title="",
        yaxis_title="Share of total (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=BRAND_TEXT),
        ),
    )
    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(tickfont=dict(color=BRAND_TEXT))
    fig.update_yaxes(tickfont=dict(color=BRAND_TEXT))
    return fig



def write_chart(fig: go.Figure, filename: str, *, transparent: bool = False) -> str:
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        tmpl = fig.layout.template
        try:
            if tmpl is not None:
                tmpl_json = tmpl.to_plotly_json()
                tmpl_json.setdefault("layout", {})["paper_bgcolor"] = "rgba(0,0,0,0)"
                tmpl_json["layout"]["plot_bgcolor"] = "rgba(0,0,0,0)"
                fig.layout.template = go.layout.Template(tmpl_json)
        except Exception:
            pass
    output_path = OUTPUT_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def copy_static_assets() -> None:
    if RES_DIR.exists():
        target = OUTPUT_DIR / "res"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(RES_DIR, target)


def build_dashboard(components: Sequence[Dict[str, object]]) -> None:
    dashboard_path = OUTPUT_DIR / "index.html"
    section_html: List[str] = []
    for comp in components:
        title = comp.get("title")
        html = comp["html"]
        classes = "card"
        extra_class = comp.get("classes")
        if extra_class:
            classes += f" {extra_class}"
        show_title = comp.get("show_title", True)
        heading = f"<h2>{title}</h2>" if show_title and title else ""
        section_html.append(
            f"""
            <section class=\"{classes}\">
                {heading}
                <div class=\"embed\">{html}</div>
            </section>
            """
        )

    kpi_style_block = KPI_CARD_STYLE.replace("\n", "\n            ")
    body_background = PAGE_GRADIENT_CSS if ENABLE_PAGE_GRADIENT else BRAND_BACKGROUND

    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Adopting Bitcoin 2024 – Lightning Adoption Report</title>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap\" rel=\"stylesheet\" />
        <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
        <style>
            :root {{
                color-scheme: dark;
                --page-bg: {BRAND_BACKGROUND};
                --surface: {BRAND_SURFACE};
                --panel: {BRAND_PANEL};
                --border: {BRAND_BORDER};
                --accent: {BRAND_ACCENT};
                --accent-2: {BRAND_MAGENTA};
                --electric: {BRAND_ELECTRIC};
                --text: {BRAND_TEXT};
                --muted: {BRAND_MUTED};
                font-family: {BRAND_FONT};
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                min-height: 100vh;
                background: {body_background};
                color: var(--text);
                padding: 40px 32px 60px;
            }}
            .page-shell {{
                max-width: 1280px;
                margin: 0 auto;
            }}
            header.page-header {{
                display: grid;
                gap: 24px;
                grid-template-columns: minmax(140px, 200px) 1fr;
                align-items: center;
                margin-bottom: 48px;
            }}
            header.page-header .logo-tile {{
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 12px;
                padding: 0;
            }}
            header.page-header .logo-tile img {{
                width: 100%;
                height: auto;
            }}
            header.page-header .headline {{
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}
            .eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.32em;
                font-size: 0.72rem;
                color: var(--accent);
                font-weight: 600;
            }}
            h1 {{
                margin: 0;
                font-size: clamp(2.2rem, 3vw, 2.9rem);
                font-weight: 700;
                line-height: 1.1;
                margin-bottom: 6px;
            }}
            p.subtitle {{
                margin: 0;
                font-size: 1.05rem;
                color: var(--muted);
                max-width: 720px;
            }}
            .chips {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 8px;
            }}
            .chip {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(255, 88, 167, 0.15);
                color: var(--text);
                font-size: 0.85rem;
                font-weight: 500;
                border: 1px solid rgba(255, 88, 167, 0.45);
            }}
            main {{
                display: grid;
                grid-auto-flow: row;
                row-gap: 92px;
                column-gap: 32px;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }}
            section.card {{
                background: var(--panel);
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 24px;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
                backdrop-filter: blur(12px);
                display: flex;
                flex-direction: column;
                gap: 16px;
                overflow: visible;
            }}
            section.card h2 {{
                font-size: 1.18rem;
                margin: 0;
                color: var(--accent);
                letter-spacing: 0.01em;
            }}
            section.card .embed {{
                flex: 1 1 auto;
                display: flex;
            }}
            section.card .embed > div {{
                width: 100%;
            }}
            section.card.full-width {{
                grid-column: 1 / -1;
            }}
            section.card.no-chrome {{
                background: transparent;
                border: none;
                box-shadow: none;
                padding: 0;
            }}
            section.card.no-chrome h2 {{
                color: var(--text);
                font-size: 1.05rem;
                margin-bottom: 12px;
            }}
            section.card.no-chrome .embed {{
                border-radius: 0;
                overflow: visible;
            }}
            section.card.no-chrome .embed > div {{
                border-radius: 0;
            }}
            .story-card.no-chrome .story-block {{
                background: transparent;
                border: none;
                box-shadow: none;
                padding: 0;
                border-radius: 0;
                margin: 0;
            }}
            .story-card.no-chrome .story-block h3 {{
                color: var(--accent);
            }}
            .story-card.no-chrome .story-block p {{
                color: var(--text);
            }}
            section.card.span-1 {{
                grid-column: span 1;
            }}
            section.card.span-2 {{
                grid-column: span 2;
            }}
            .card.hide-sm {{
                display: flex;
            }}
            .card.show-sm {{
                display: none;
            }}
            .card.md-full {{
                /* defaults maintain auto layout */
            }}
            .card.md-half {{
                /* defaults maintain auto layout */
            }}
            .card.md-span-1 {{
                /* defaults maintain auto layout */
            }}
            .story-block {{
                background: rgba(14, 16, 48, 0.82);
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
                padding: 24px 28px;
                display: flex;
                flex-direction: column;
                gap: 18px;
            }}
            .story-block h3 {{
                margin: 0;
                color: var(--accent);
                font-size: 1.3rem;
            }}
            .story-block p {{
                margin: 0;
                color: var(--text);
                line-height: 1.6;
            }}
            section.card.is-kpi {{
                background: transparent;
                border: none;
                box-shadow: none;
                padding: 0;
                min-height: unset;
            }}
            section.card.is-kpi h2 {{
                font-size: 1rem;
                color: var(--accent);
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}
            section.card.is-kpi .embed {{
                align-items: stretch;
            }}
            section.card.is-kpi .embed > div {{
                display: flex;
                width: 100%;
            }}
            .plotly .modebar {{
                top: 12px;
                right: 12px;
            }}
            {kpi_style_block}
            footer {{
                margin-top: 28px;
                font-size: 0.85rem;
                color: var(--muted);
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }}
            footer a {{
                color: var(--accent);
                text-decoration: none;
            }}
            @media (max-width: 1100px) {{
                main {{
                    row-gap: 72px;
                }}
            }}
            @media (min-width: 721px) and (max-width: 900px) {{
                .card.md-half {{
                    grid-column: span 1;
                }}
                .card.md-full {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-1, section.card.md-span-1 {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-2 {{
                    grid-column: span 2;
                }}
            }}
            @media (max-width: 900px) {{
                header.page-header {{
                    grid-template-columns: 1fr;
                    text-align: center;
                }}
                header.page-header .logo-tile {{
                    max-width: 200px;
                    margin: 0 auto;
                }}
                header.page-header .headline {{
                    align-items: center;
                }}
                p.subtitle {{
                    text-align: center;
                }}
                main {{
                    row-gap: 56px;
                    grid-template-columns: repeat(2, minmax(280px, 1fr));
                }}
                .card.md-half {{
                    grid-column: span 1;
                }}
                .card.md-full {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-1, section.card.md-span-1 {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-2 {{
                    grid-column: span 2;
                }}
            }}
            @media (max-width: 720px) {{
                main {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    column-gap: 18px;
                    row-gap: 40px;
                }}
                section.card {{
                    grid-column: 1 / -1;
                }}
                section.card.sm-full {{
                    grid-column: 1 / -1;
                }}
                section.card.sm-half {{
                    grid-column: span 1;
                }}
                section.card.is-kpi {{
                    margin: 0;
                }}
                .plotly .modebar {{
                    top: 8px;
                    right: 8px;
                    transform: scale(0.85);
                    transform-origin: top right;
                }}
                .card.md-half {{
                    grid-column: span 1;
                }}
                .card.md-full {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-1 {{
                    grid-column: 1 / -1;
                }}
                section.card.md-span-1 {{
                    grid-column: 1 / -1;
                }}
                .card.md-span-2 {{
                    grid-column: span 2;
                }}
                .card.hide-sm {{
                    display: none !important;
                }}
                .card.show-sm {{
                    display: flex !important;
                }}
            }}
            @media (max-width: 520px) {{
                main {{
                    grid-template-columns: 1fr;
                    row-gap: 36px;
                    column-gap: 0;
                }}
                section.card.sm-half {{
                    grid-column: 1 / -1;
                }}
                .plotly .modebar {{
                    top: 6px;
                    right: 6px;
                    transform: scale(0.8);
                }}
                .card.show-sm {{
                    display: flex !important;
                }}
                .card.md-full {{
                    grid-column: 1 / -1;
                }}
            }}
        </style>
    </head>
    <body>
        <div class=\"page-shell\">
            <header class=\"page-header\">
                <div class=\"logo-tile\">
                    <img src=\"res/main-logo-neon.png\" alt=\"Adopting Bitcoin\" />
                </div>
                <div class=\"headline\">
                    <span class=\"eyebrow\">Lightning Adoption Report Perspective (LARP)</span>
                    <h1>Adopting Bitcoin 2024</h1>
                    <p class=\"subtitle\">Showcasing how attendees embraced bitcoin and lightning payments across the conference floor, side events, and compared with peer gatherings.</p>
                    <div class=\"chips\">
                        <span class=\"chip\">⚡ 2.47 transactions per attendee</span>
                        <span class=\"chip\">🎨 1.14 BTC spent on art</span>
                    </div>
                </div>
            </header>
            <main>
                {''.join(section_html)}
            </main>
            <footer>
                <span>Data sources: conference overview + AB24 lightning sales detail.</span>
                <span>Charts generated with Plotly · branded for Adopting Bitcoin.</span>
            </footer>
        </div>
    </body>
    </html>
    """
    dashboard_path.write_text(dashboard_html, encoding="utf-8")


def main() -> None:
    overview_df = load_overview()
    detail_df = load_details()
    activity_detail = detail_df[detail_df["CategoryGroup"] != "Sponsors/Tickets"].copy()

    ab24 = overview_df[(overview_df["Event"] == "Adopting Bitcoin") & (overview_df["Year"] == 2024)]
    if ab24.empty:
        raise RuntimeError("Could not locate Adopting Bitcoin 2024 row in overview data.")
    ab24_row = ab24.iloc[0]

    kpis = create_ab24_kpis(ab24_row)
    top_kpis = kpis[:3]
    trailing_kpis = kpis[3:]
    sankey_fig = create_ab24_sankey(activity_detail)
    venue_split_fig = create_on_off_site_split(activity_detail)

    comparison_df = prepare_comparison_df(overview_df)
    tx_bar_fig = create_transactions_bar_fig(comparison_df, show_labels=True)
    tx_bar_fig_compact = create_transactions_bar_fig(comparison_df, show_labels=False)
    scatter_fig = create_peer_scatter_fig(comparison_df)

    components: List[Dict[str, object]] = []

    for idx, kpi in enumerate(top_kpis):
        responsive_class = "sm-full" if idx == 0 else "sm-half"
        md_class = "md-half" if idx in (0, 1) else ("md-full" if idx == 2 else "")
        classes = " ".join(filter(None, ["is-kpi", responsive_class, md_class]))
        components.append(
            {
                "title": kpi["title"],
                "html": write_kpi_component(kpi),
                "classes": classes,
            }
        )

    components.append(
        {
            "title": "Lightning Flow Across Activities",
            "html": write_chart(sankey_fig, "ab24_sankey.html", transparent=True),
            "classes": "full-width no-chrome sm-full",
        }
    )

    components.append(
        {
            "title": "Where Lightning Usage Happened",
            "html": write_chart(venue_split_fig, "ab24_venue_split.html", transparent=True),
            "classes": "span-1 no-chrome sm-full md-span-1",
        }
    )
    components.append(
        {
            "title": "How attendees lit up the conference",
            "html": LIGHTNING_STORY_HTML,
            "classes": "span-2 story-card no-chrome sm-full md-span-1",
            "show_title": False,
        }
    )

    components.append(
        {
            "title": "Transactions at Bitcoin Conferences",
            "html": write_chart(tx_bar_fig, "comparison_transactions.html", transparent=True),
            "classes": "span-2 no-chrome sm-full md-span-1 hide-sm",
        }
    )
    components.append(
        {
            "title": "Transactions at Bitcoin Conferences",
            "html": write_chart(tx_bar_fig_compact, "comparison_transactions_compact.html", transparent=True),
            "classes": "span-2 no-chrome sm-full show-sm",
        }
    )
    components.append(
        {
            "title": "Lightning Throughput",
            "html": write_chart(scatter_fig, "comparison_scale_vs_throughput.html", transparent=True),
            "classes": "span-1 no-chrome sm-full md-span-1",
        }
    )

    remaining_trailing: List[Dict[str, object]] = []
    tx_per_attendee_kpi = None
    for kpi in trailing_kpis:
        if kpi.get("slug") == "tx-per-attendee" and tx_per_attendee_kpi is None:
            tx_per_attendee_kpi = kpi
        else:
            remaining_trailing.append(kpi)

    if tx_per_attendee_kpi:
        components.append(
            {
                "title": "",
                "html": TRANSACTION_STORY_HTML,
                "classes": "span-2 story-card no-chrome sm-full",
                "show_title": False,
            }
        )
        components.append(
            {
                "title": tx_per_attendee_kpi["title"],
                "html": write_kpi_component(tx_per_attendee_kpi),
                "classes": "is-kpi span-1 sm-full",
            }
        )

    for kpi in remaining_trailing:
        components.append(
            {
                "title": kpi["title"],
                "html": write_kpi_component(kpi),
                "classes": "is-kpi sm-full",
            }
        )

    copy_static_assets()
    build_dashboard(components)
    print(json.dumps({"output_dir": str(OUTPUT_DIR)}))


if __name__ == "__main__":
    main()
