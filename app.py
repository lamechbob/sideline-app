# South Broward Football – Streamlit App (2025)
# -------------------------------------------------
# Key updates in this version:
# - Column normalizer handles common variants (receptions -> catches, punting_yards -> punt_yards, etc)
# - total_tackles can include sacks (toggle via secrets)
# - Week picker defaults to latest week
# - Better jersey number formatting and player labels
# - "Last updated" banner from game_date
# - Sidebar spacing tuned + centered logo
# - Season Leaders rendered as tables (no index)
# - Weekly View: Pass Deflections next to Interceptions; Rush Attempts from "Rush" events when needed;
#   Rushing Average (yds/att) & Receiving Average (yds/catch); Targets fixed to only "Pas Target" + "Catch"
# - Player Details: Added Solo Tackles, Assisted Tackles, Deflections, Rushing Average, Receiving Average in Season Totals
#   and included Assisted Tackles + Deflections + per-game rushing/receiving averages in Game Log
# -------------------------------------------------

import io
import os
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="South Broward Football – 2025", layout="wide")

# --- Sidebar spacing (roomy) + centered logo + black first column in tables ---
st.markdown(
    """
    <style>
      /* Sidebar padding */
      section[data-testid="stSidebar"] > div.block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
      }

      /* Center the logo wrapper (covers multiple Streamlit builds) */
      section[data-testid="stSidebar"] [data-testid="stImage"],
      section[data-testid="stSidebar"] .stImage,
      section[data-testid="stSidebar"] figure.stImage {
        display: flex !important;
        justify-content: center !important;
      }
      /* Ensure the image itself is centered */
      section[data-testid="stSidebar"] [data-testid="stImage"] img,
      section[data-testid="stSidebar"] .stImage img,
      section[data-testid="stSidebar"] figure.stImage img {
        margin: 0 auto !important;
        display: block !important;
      }
      /* Slightly trim space under images */
      section[data-testid="stSidebar"] .stImage { margin-bottom: 0.35rem; }

      /* Compact rules/headings/text */
      section[data-testid="stSidebar"] hr { margin: 0.45rem 0; }
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 { margin: 0.15rem 0 0.4rem 0; }
      section[data-testid="stSidebar"] p { margin: 0.2rem 0; }

      /* Make the first visible column (Player) render as black in st.table (fallback) */
      [data-testid="stTable"] table tbody tr td:first-child { color: #000 !important; }
      [data-testid="stTable"] table thead tr th:first-child { color: #000 !important; }
      /* Hide the index column in st.table entirely (row headers + top-left corner) */
      [data-testid="stTable"] table th.row_heading,
      [data-testid="stTable"] table td.row_heading { display: none !important; }
      [data-testid="stTable"] table th.blank { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Configuration via Secrets
# ============================
MODE = st.secrets.get("DATA_MODE", "PUBLIC_S3")  # "PUBLIC_S3" or "PRIVATE_S3"
PUBLIC_S3_URL = st.secrets.get("PUBLIC_S3_URL", "")
AWS_REGION = st.secrets.get("AWS_REGION", "")
S3_BUCKET = st.secrets.get("S3_BUCKET", "")
S3_KEY = st.secrets.get("S3_KEY", "")

# Logo width (px). Override with LOGO_WIDTH in secrets if desired.
try:
    LOGO_WIDTH = int(st.secrets.get("LOGO_WIDTH", 180))
except Exception:
    LOGO_WIDTH = 180

# Whether to count sacks inside total tackles when recomputing totals.
INCLUDE_SACKS_IN_TACKLES = bool(st.secrets.get("INCLUDE_SACKS_IN_TACKLES", False))

# Fixed context
FIXED_SEASON = 2025
FIXED_TEAM_NAME = "South Broward"

# ============================
# Expectations / Schema
# ============================
REQUIRED_BASE = {
    "season_year", "week_number", "game_date", "team_id", "team_name",
    "player_id", "first_name", "last_name"
}

STAT_COLS = [
    "passing_yards",
    "passing_completions", "passing_attempts", "passing_tds",
    "rush_attempts", "rushing_yards", "rushing_tds",
    "targets", "catches", "receiving_yards", "receiving_tds",
    "solo_tackles", "assisted_tackles", "total_tackles",
    "sacks", "tackles_for_loss",   # <-- add this line
    "deflections", "interceptions", "defensive_tds", "safeties",
    "fg_attempts", "fg_made", "pat_attempts", "pat_made",
    "punts", "punt_yards",
    "kick_returns", "kick_return_yards",
    "punt_returns", "punt_return_yards",
]


COL_LABELS: Dict[str, str] = {
    "passing_yards": "Passing Yards",
    "passing_completions": "Pass Completions",
    "passing_attempts": "Pass Attempts",
    "passing_tds": "Pass TDs",
    "rush_attempts": "Rush Attempts",
    "rushing_yards": "Rushing Yards",
    "rushing_tds": "Rushing TDs",
    "rushing_avg": "Rushing Average",
    "targets": "Targets",
    "catches": "Receptions",
    "receiving_yards": "Receiving Yards",
    "receiving_tds": "Receiving TDs",
    "receiving_avg": "Receiving Average",
    "solo_tackles": "Solo Tackles",
    "assisted_tackles": "Assisted Tackles",
    "total_tackles": "Total Tackles",
    "sacks": "Sacks",
    "tackles_for_loss": "Tackles for Loss",  # <-- add this
    "deflections": "Pass Deflections",
    "interceptions": "Interceptions",
    "defensive_tds": "Defensive TDs",
    "safeties": "Safeties",
    "fg_attempts": "FG Attempts",
    "fg_made": "FG Made",
    "pat_attempts": "PAT Attempts",
    "pat_made": "PAT Made",
    "punts": "Punts",
    "punt_yards": "Punt Yards",
    "kick_returns": "Kick Returns",
    "kick_return_yards": "Kick Return Yards",
    "punt_returns": "Punt Returns",
    "punt_return_yards": "Punt Return Yards",
    "avg_punt_yards": "Avg Punt Yds",
}
NUMERIC_COLS = list(COL_LABELS.keys())

# Common variants that we will normalize
NORMALIZE_MAP = {
    "receptions": "catches",
    "punting_yards": "punt_yards",
    "field_goals_made": "fg_made",
    "field_goals_attempts": "fg_attempts",
    "extra_points_made": "pat_made",
    "extra_points_attempts": "pat_attempts",
    "pass_completions": "passing_completions",
    "pass_attempts": "passing_attempts",
    "pass_tds": "passing_tds",
    "receiving_td": "receiving_tds",
    "rushing_td": "rushing_tds",
    "kick_return_yard": "kick_return_yards",
    "punt_return_yard": "punt_return_yards",
}

# ============================
# Data Loading
# ============================
@st.cache_data(ttl=300)
def _load_public_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


@st.cache_data(ttl=300)
def _load_private_s3(bucket: str, key: str, region: str) -> pd.DataFrame:
    import boto3
    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)
    df = df.rename(columns={k: v for (k, v) in NORMALIZE_MAP.items() if k in df.columns})
    return df


@st.cache_data(ttl=300)
def load_summary_df() -> pd.DataFrame:
    if MODE == "PRIVATE_S3":
        df = _load_private_s3(S3_BUCKET, S3_KEY, AWS_REGION)
    else:
        if not PUBLIC_S3_URL:
            st.error("PUBLIC_S3_URL secret is missing.")
            st.stop()
        df = _load_public_csv(PUBLIC_S3_URL)

    df = _normalize_columns(df)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    missing = REQUIRED_BASE - set(df.columns)
    if missing:
        st.error(f"Missing columns in summary CSV: {sorted(missing)}")
        st.stop()

    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["player_name"] = (df["first_name"].fillna("").astype(str).str.strip() + " " +
                         df["last_name"].fillna("").astype(str).str.strip()).str.strip()

    for c in ["height_in", "weight_lb", "jersey_number"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "position" in df.columns:
        df["position"] = df["position"].astype(str)

    if INCLUDE_SACKS_IN_TACKLES:
        if "solo_tackles" in df.columns and "assisted_tackles" in df.columns:
            sacks = df["sacks"] if "sacks" in df.columns else 0
            df["total_tackles"] = (
                pd.to_numeric(df["solo_tackles"], errors="coerce").fillna(0)
                + pd.to_numeric(df["assisted_tackles"], errors="coerce").fillna(0)
                + pd.to_numeric(sacks, errors="coerce").fillna(0)
            )
    return df


@st.cache_data(ttl=300)
def load_roster_df() -> pd.DataFrame:
    """No external roster file; return empty so the app uses inline roster fields."""
    return pd.DataFrame(columns=["player_id", "height_in", "weight_lb", "position", "jersey_number"])

# ============================
# Helpers
# ============================

def aggregate_player_totals(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["season_year", "team_id", "team_name", "player_id", "player_name"]
    agg_dict = {c: "sum" for c in STAT_COLS if c in df.columns}
    if "jersey_number" in df.columns:
        agg_dict["jersey_number"] = "max"
    out = df.groupby(group_cols, dropna=False, as_index=False).agg(agg_dict)
    return out


def humanize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: COL_LABELS[c] for c in df.columns if c in COL_LABELS}
    return df.rename(columns=rename_map)


def _label_with_jersey(name: str, jersey) -> str:
    # Show jersey if it’s a real number, including 0
    try:
        import pandas as pd
        if jersey is None or (isinstance(jersey, float) and pd.isna(jersey)):
            return name
        j = int(float(jersey))
        # Football allows 0; negative is not a real jersey so skip
        if j >= 0:
            return f"#{j} {name}"
    except Exception:
        pass
    return name


def top_k(agg: pd.DataFrame, stat: str, k: int = 3) -> pd.DataFrame:
    if stat not in agg.columns:
        return pd.DataFrame(columns=["Player", COL_LABELS.get(stat, stat)])
    cols = ["player_name", stat]
    if "jersey_number" in agg.columns:
        cols.append("jersey_number")
    tmp = agg[cols].copy()
    tmp = tmp.sort_values(stat, ascending=False).head(k)

    if "jersey_number" in tmp.columns:
        tmp["Player"] = tmp.apply(lambda r: _label_with_jersey(str(r["player_name"]), r["jersey_number"]), axis=1)
    else:
        tmp["Player"] = tmp["player_name"]

    tmp = tmp[["Player", stat]].copy()
    tmp = tmp.rename(columns={stat: COL_LABELS.get(stat, stat)})
    return tmp


def leader_table(df: pd.DataFrame, stat: str):
    """Return a Pandas Styler for st.table: black Player names, no index, clean numeric format."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Player", COL_LABELS.get(stat, stat)])

    label = COL_LABELS.get(stat, stat)
    dfi = df[["Player", label]].copy()  # keep Player as a normal column

    fmt = "{:,.1f}" if stat == "avg_punt_yards" else "{:,.0f}"
    try:
        sty = dfi.style.format({label: fmt}).set_properties(subset=["Player"], **{"color": "black"})
        # Compatibility: some envs need hide_index(), others hide(axis="index")
        if hasattr(sty, "hide_index"):
            sty = sty.hide_index()
        else:
            sty = sty.hide(axis="index")
        return sty
    except Exception:
        # Fallback if Styler fails (rare); keep as plain DataFrame (index may appear but CSS will hide it)
        return dfi


def fmt_height_inches(h) -> str:
    """Format height in inches as feet'inches" (e.g., 72 -> 6' 0")."""
    try:
        if h is None or (isinstance(h, float) and pd.isna(h)):
            return "—"
        h_int = int(round(float(h)))
        feet = h_int // 12
        inches = h_int % 12
        return f"{feet}' {inches}\""
    except Exception:
        return "—"


def _event_text_columns(df: pd.DataFrame) -> list:
    """Return plausible text columns that may carry per-play/event names."""
    return [c for c in ["stat_name", "stat", "event", "event_name", "action", "type", "category"] if c in df.columns]


def weekly_event_counts(vf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a (week-filtered) frame `vf`, compute:
      - rush_attempts_calc: # rows where any text col equals 'Rush'
      - targets_calc:       # rows where text col is exactly 'Pas Target' or 'Catch'
    Returns (rush_counts, targets_counts) keyed by player.
    """
    gcols = ["player_id", "player_name"]
    text_cols = _event_text_columns(vf)
    rush_counts = pd.DataFrame(columns=gcols + ["rush_attempts_calc"])
    target_counts = pd.DataFrame(columns=gcols + ["targets_calc"])

    if not text_cols:
        return rush_counts, target_counts

    tmp = vf.copy()
    for c in text_cols:
        tmp[c] = tmp[c].astype(str)

    # Rushing Attempts: exact 'Rush' (case-insensitive) in any text column
    rush_mask = False
    for c in text_cols:
        rush_mask = rush_mask | tmp[c].str.lower().eq("rush")

    if getattr(rush_mask, "any", lambda: False)():
        rush_counts = (
            tmp.loc[rush_mask]
               .groupby(gcols)
               .size()
               .reset_index(name="rush_attempts_calc")
        )

    # Targets: ONLY 'Pas Target' and 'Catch' (case-insensitive)
    tgt_mask = False
    target_terms = {"pas target", "catch"}
    for c in text_cols:
        tgt_mask = tgt_mask | tmp[c].str.lower().isin(target_terms)

    if getattr(tgt_mask, "any", lambda: False)():
        target_counts = (
            tmp.loc[tgt_mask]
               .groupby(gcols)
               .size()
               .reset_index(name="targets_calc")
        )

    return rush_counts, target_counts

# ============================
# Player Details helper (updated)
# ============================

def show_player_detail(player_id: str, season_year: int, full_df: pd.DataFrame, roster_df: pd.DataFrame):
    """Render a player's season summary + game log for the given season.

    Updates:
      - Season Totals now include Assisted Tackles, Pass Deflections, Rushing Average, Receiving Average.
      - Game Log now includes Assisted Tackles and Pass Deflections, plus per-game rushing/receiving averages.
    """
    pdf = full_df[(full_df["player_id"] == player_id) & (full_df["season_year"] == season_year)].copy()
    if pdf.empty:
        st.info("No data found for this player.")
        return

    name = pdf["player_name"].iloc[0]
    team = pdf["team_name"].iloc[0]

    # Totals across the season for the selected player
    totals = {}
    for c in STAT_COLS:
        if c in pdf.columns:
            totals[c] = float(pd.to_numeric(pdf[c], errors="coerce").fillna(0).sum())

    # Derived season averages (divide-by-zero safe)
    def _safe_div(n, d):
        try:
            n = float(n)
            d = float(d)
            return n / d if d else 0.0
        except Exception:
            return 0.0

    totals["rushing_avg"] = round(_safe_div(totals.get("rushing_yards", 0.0), totals.get("rush_attempts", 0.0)), 1)
    totals["receiving_avg"] = round(_safe_div(totals.get("receiving_yards", 0.0), totals.get("catches", 0.0)), 1)

    # Prefer external roster; fall back to inline columns in the summary view
    height = weight = position = jersey = None
    r = roster_df[roster_df["player_id"] == player_id] if not roster_df.empty else pd.DataFrame()
    if not r.empty:
        height = r.get("height_in", pd.Series([None])).iloc[0]
        weight = r.get("weight_lb", pd.Series([None])).iloc[0]
        position = r.get("position", pd.Series([""])).iloc[0]
        jersey = r.get("jersey_number", pd.Series([None])).iloc[0]
    else:
        if "height_in" in pdf.columns and pdf["height_in"].notna().any():
            height = pd.to_numeric(pdf["height_in"], errors="coerce").dropna().iloc[0]
        if "weight_lb" in pdf.columns and pdf["weight_lb"].notna().any():
            weight = pd.to_numeric(pdf["weight_lb"], errors="coerce").dropna().iloc[0]
        if "position" in pdf.columns and pdf["position"].notna().any():
            position = str(pdf["position"].dropna().astype(str).iloc[0])
        if "jersey_number" in pdf.columns and pdf["jersey_number"].notna().any():
            jersey = pd.to_numeric(pdf["jersey_number"], errors="coerce").dropna().iloc[0]

    cols = st.columns([2, 1, 1, 1, 1])
    with cols[0]:
        st.subheader(name)
        st.caption(team)
    with cols[1]:
        st.metric("Height", value=fmt_height_inches(height))
    with cols[2]:
        st.metric("Weight (lb)", value="—" if pd.isna(weight) or weight is None else int(weight))
    with cols[3]:
        st.metric("Position", value=position if position else "—")
    with cols[4]:
        st.metric("Jersey #", value="—" if pd.isna(jersey) or jersey is None else int(jersey))

    key_sections = [
        ("Passing",   ["passing_yards", "passing_completions", "passing_attempts", "passing_tds"]),
        ("Rushing",   ["rush_attempts", "rushing_yards", "rushing_avg", "rushing_tds"]),
        ("Receiving", ["targets", "catches", "receiving_yards", "receiving_avg", "receiving_tds"]),
        ("Defense", ["total_tackles", "solo_tackles", "assisted_tackles",
                     "sacks", "tackles_for_loss",  # <-- add here
                     "interceptions", "deflections", "defensive_tds", "safeties"]),
        ("Kicking",   ["fg_attempts", "fg_made", "pat_attempts", "pat_made", "punts"]),
        ("Returns",   ["kick_returns", "kick_return_yards", "punt_returns", "punt_return_yards"]),
    ]

    st.subheader("Season Totals")

    for section, cols_list in key_sections:
        # Consider derived metrics present if we computed them into `totals`
        present = [c for c in cols_list if (c in pdf.columns) or (c in totals)]
        if not present:
            continue
        st.subheader(f"{section}")
        metrics = st.columns(len(present))
        for i, c in enumerate(present):
            label = COL_LABELS.get(c, c.replace("_", " ").title())
            val = totals.get(c, 0)
            if c in {"rushing_avg", "receiving_avg", "avg_punt_yards"}:
                metrics[i].metric(label, f"{float(val):.1f}")
            else:
                metrics[i].metric(label, int(float(val)))

    st.markdown("### Game log")

    # Per-game averages for display
    if "rushing_yards" in pdf.columns and "rush_attempts" in pdf.columns:
        pdf["rushing_avg"] = (
            pd.to_numeric(pdf["rushing_yards"], errors="coerce").fillna(0)
            / pd.to_numeric(pdf["rush_attempts"], errors="coerce").replace(0, pd.NA)
        ).fillna(0).round(1)
    if "receiving_yards" in pdf.columns and "catches" in pdf.columns:
        pdf["receiving_avg"] = (
            pd.to_numeric(pdf["receiving_yards"], errors="coerce").fillna(0)
            / pd.to_numeric(pdf["catches"], errors="coerce").replace(0, pd.NA)
        ).fillna(0).round(1)

    display_cols = [c for c in [
        "game_date", "week_number",
        "passing_completions", "passing_attempts", "passing_tds", "passing_yards",
        "rush_attempts", "rushing_yards", "rushing_tds", "rushing_avg",
        "catches", "receiving_yards", "receiving_tds", "receiving_avg",
        "total_tackles", "solo_tackles", "assisted_tackles", "sacks", "interceptions", "deflections",
        "fg_made", "pat_made",
        "punts", "punt_yards",
        "kick_returns", "kick_return_yards", "punt_returns", "punt_return_yards"
    ] if c in pdf.columns]

    pdf_disp = pdf.sort_values(["game_date", "week_number"], na_position="last")[display_cols]
    st.dataframe(humanize_cols(pdf_disp), use_container_width=True)

# ============================
# Load data
# ============================
summary_df = load_summary_df()
roster_df = load_roster_df()

# Keep only 2025 + South Broward if present
f = summary_df[summary_df["season_year"] == FIXED_SEASON].copy()
if "team_name" in f.columns and FIXED_TEAM_NAME in set(f["team_name"].unique()):
    f = f[f["team_name"] == FIXED_TEAM_NAME]

st.title("South Broward Football – 2025")
if "game_date" in f.columns and f["game_date"].notna().any():
    last_dt = pd.to_datetime(f["game_date"], errors="coerce").dropna().max()
    st.caption(f"As of {last_dt.date().isoformat()} • This app provides real-time and season-to-date football stats"
               f", leaderboards, and player details , making it easy to track team performance and individual "
               f"achievements throughout the season.")

else:
    st.caption("This App provides real-time and season-to-date football stats, leaderboards, and player details"
               ", making it easy to track team performance and individual achievements throughout the season.")

# ============================
# Sidebar (Navigation)
# ============================
# Logo (prefer local path, then URL)
logo_shown = False
logo_path = st.secrets.get("LOGO_PATH", "")
candidate_paths = [p for p in [logo_path, "images/sbhs_logo.png", "sideline-app/images/sbhs_logo.png"] if p]
for p in candidate_paths:
    try:
        if p and os.path.exists(p):
            st.sidebar.image(p, width=LOGO_WIDTH)
            logo_shown = True
            break
    except Exception:
        pass

if not logo_shown:
    logo_url = st.secrets.get("LOGO_URL", "")
    if logo_url:
        st.sidebar.image(logo_url, width=LOGO_WIDTH)
        logo_shown = True

st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
NAV_ITEMS = ["Season Leaders", "Weekly View", "Player Details"]
nav = st.sidebar.selectbox("", NAV_ITEMS, label_visibility="collapsed", key="nav")

# ============================
# Season Leaders view (tables)
# ============================
if nav == "Season Leaders":
    agg = aggregate_player_totals(f)

    # Compute derived special-teams metrics
    if "avg_punt_yards" not in agg.columns:
        if "punt_yards" in agg.columns and "punts" in agg.columns:
            denom = agg["punts"].replace({0: pd.NA})
            agg["avg_punt_yards"] = (pd.to_numeric(agg["punt_yards"], errors="coerce").fillna(0) / denom).fillna(0)

    st.subheader("Season Leaders")
    colA, colB = st.columns(2)

    # Leaderboards
    passing_top = top_k(agg, "passing_yards", 3) if "passing_yards" in agg.columns else pd.DataFrame()
    rushing_top = top_k(agg, "rushing_yards", 3)
    receiving_top = top_k(agg, "receiving_yards", 3)
    tacklers_top = top_k(agg, "total_tackles", 3)
    defenders_top = top_k(agg, "interceptions", 3)
    sackers_top = top_k(agg, "sacks", 3)
    fg_top = top_k(agg, "fg_made", 3)
    pat_top = top_k(agg, "pat_made", 3)
    punts_avg_top = top_k(agg, "avg_punt_yards", 3) if "avg_punt_yards" in agg.columns else pd.DataFrame()

    with colA:
        st.markdown("#### Passing")
        if not passing_top.empty:
            st.table(leader_table(passing_top, "passing_yards"))
        else:
            st.info("Add `passing_yards` to the summary CSV to populate this leaderboard.")

        st.markdown("#### Receiving")
        st.table(leader_table(receiving_top, "receiving_yards"))

        st.markdown("#### Interceptions")
        st.table(leader_table(defenders_top, "interceptions"))

        st.markdown("#### Kicking")
        st.table(leader_table(fg_top, "fg_made"))

        st.markdown("#### Extra Points")
        st.table(leader_table(pat_top, "pat_made"))

    with colB:
        st.markdown("#### Rushing")
        st.table(leader_table(rushing_top, "rushing_yards"))

        st.markdown("#### Tackles")
        st.table(leader_table(tacklers_top, "total_tackles"))

        st.markdown("#### Sacks")
        st.table(leader_table(sackers_top, "sacks"))

        st.markdown("#### Punting")
        if not punts_avg_top.empty:
            st.table(leader_table(punts_avg_top, "avg_punt_yards"))
        else:
            st.info("Provide `punt_yards` and `punts` to compute average yards per punt.")

# ============================
# Weekly View
# ============================
elif nav == "Weekly View":
    weeks = sorted([w for w in f["week_number"].dropna().unique()])
    if weeks:
        default_index = weeks.index(max(weeks))
        selected_week = st.selectbox("Week", weeks, index=default_index)
        vf = f[f["week_number"] == selected_week].copy()
    else:
        st.info("No week numbers found.")
        st.stop()

    if vf.empty:
        st.info("No data for the selected week.")
        st.stop()

    # Sum up numerics by player for the week
    players_totals = aggregate_player_totals(vf)

    # Merge in event-derived counts (rush attempts and fixed targets) if we can infer them
    rush_counts, target_counts = weekly_event_counts(vf)
    if not rush_counts.empty:
        players_totals = players_totals.merge(rush_counts, on=["player_id", "player_name"], how="left")
    if not target_counts.empty:
        players_totals = players_totals.merge(target_counts, on=["player_id", "player_name"], how="left")

    # Choose the best 'Rush Attempts'
    if "rush_attempts" in players_totals.columns:
        players_totals["rush_attempts"] = pd.to_numeric(players_totals["rush_attempts"], errors="coerce").fillna(0)
    else:
        players_totals["rush_attempts"] = 0

    if "rush_attempts_calc" in players_totals.columns:
        players_totals["rush_attempts"] = players_totals["rush_attempts"].mask(
            players_totals["rush_attempts"] <= 0, players_totals["rush_attempts_calc"]
        )

    players_totals["rush_attempts"] = players_totals["rush_attempts"].fillna(0).astype(int)

    # Fix Targets: if we computed targets_calc from events (ONLY 'Pas Target' + 'Catch'), use it; else keep numeric
    if "targets_calc" in players_totals.columns:
        fallback_targets = pd.to_numeric(players_totals.get("targets", 0), errors="coerce").fillna(0)
        players_totals["targets"] = pd.to_numeric(players_totals["targets_calc"], errors="coerce").fillna(fallback_targets).astype(int)
    else:
        if "targets" in players_totals.columns:
            players_totals["targets"] = pd.to_numeric(players_totals["targets"], errors="coerce").fillna(0).astype(int)

    # Averages (divide-by-zero safe)
    if "rushing_yards" in players_totals.columns:
        players_totals["rushing_avg"] = (
            pd.to_numeric(players_totals["rushing_yards"], errors="coerce").fillna(0)
            / players_totals["rush_attempts"].replace(0, pd.NA)
        ).fillna(0)

    if "receiving_yards" in players_totals.columns and "catches" in players_totals.columns:
        players_totals["receiving_avg"] = (
            pd.to_numeric(players_totals["receiving_yards"], errors="coerce").fillna(0)
            / pd.to_numeric(players_totals["catches"], errors="coerce").replace(0, pd.NA)
        ).fillna(0)

    # Round averages for display
    for c in ["rushing_avg", "receiving_avg"]:
        if c in players_totals.columns:
            players_totals[c] = players_totals[c].round(1)

    # Build display name with jersey if available
    if "jersey_number" in players_totals.columns:
        players_totals["display_name"] = players_totals.apply(
            lambda r: _label_with_jersey(str(r["player_name"]), r.get("jersey_number")), axis=1
        )
    else:
        players_totals["display_name"] = players_totals["player_name"]

    st.subheader(f"Week {selected_week} Totals")

    # Include new stats in the table (Deflections next to Interceptions)
    # Include new stats in the table (Deflections next to Interceptions)
    show_cols = [c for c in [
        "display_name",
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rush_attempts",
        "rushing_avg",
        "rushing_tds",
        "catches",
        "receiving_yards",
        "receiving_avg",
        "receiving_tds",
        "targets",
        "total_tackles",
        "solo_tackles",
        "assisted_tackles",
        "sacks",
        "tackles_for_loss",  # <-- add here
        "interceptions",
        "deflections",             # next to Interceptions
        "fg_attempts",             # <-- added
        "fg_made",
        "pat_attempts",            # <-- added
        "pat_made",
        "punts",
        "punt_yards",
        "kick_returns",
        "kick_return_yards",
        "punt_returns",
        "punt_return_yards",
    ] if c in players_totals.columns]


    table = players_totals[show_cols].rename(columns={"display_name": "Player"})
    st.dataframe(humanize_cols(table), use_container_width=True, hide_index=True)

# ============================
# Player Details view
# ============================
elif nav == "Player Details":
    st.subheader("Player Details")

    if f.empty:
        st.info("No players found.")
        st.stop()

    # Build latest jersey per player (by game_date), then label as "#N First Last"
    cols = ["player_id", "player_name", "game_date"]
    if "jersey_number" in f.columns:
        cols.append("jersey_number")
    pf = f[cols].dropna(subset=["player_id", "player_name"]).copy()

    pf["game_date"] = pd.to_datetime(pf["game_date"], errors="coerce")
    if "jersey_number" not in pf.columns:
        pf["jersey_number"] = pd.NA

    pf = pf.sort_values(["player_id", "game_date"], na_position="first")
    latest = (
        pf.groupby("player_id", as_index=False)
          .agg({"player_name": "last", "jersey_number": "last"})
    )

    # Sort by jersey (numeric; NaN last), then name
    latest["jersey_sort"] = pd.to_numeric(latest["jersey_number"], errors="coerce")
    latest = latest.sort_values(["jersey_sort", "player_name"], na_position="last").reset_index(drop=True)

    # Labels like "#0 Steven Wright" (uses helper that allows 0)
    label_map = {
        row.player_id: _label_with_jersey(row.player_name, row.jersey_number)
        for _, row in latest.iterrows()
    }

    options = latest["player_id"].astype(str).tolist()
    sel_pid = st.selectbox(
        "Player",
        options=options,
        format_func=lambda pid: label_map.get(pid, pid),
        key="player_details_select",
    )

    show_player_detail(sel_pid, season_year=FIXED_SEASON, full_df=summary_df, roster_df=roster_df)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.95rem;'>Deployed and Maintained by "
    "<a href='https://lamech.dev' target='_blank'>Lamech Bob-Manuel</a></div>",
    unsafe_allow_html=True,
)

# ============================
# requirements.txt (separate file)
# ----------------------------
# streamlit==1.36.0
# pandas==2.2.2
# boto3==1.34.162
