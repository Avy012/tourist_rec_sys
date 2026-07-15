from __future__ import annotations

import ast
import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# App configuration
# =========================================================
st.set_page_config(
    page_title="SmartTrip AI",
    page_icon="🌏",
    layout="wide",
)

TOP_N = 6
PROFILE_CSV_PATH = Path("data/tourist_profile_data_with_tips.csv")
INSIGHT_CSV_PATH = Path("data/Visitor_Insights_Keywords_OWI_v4.csv")
IMAGE_DIR = Path("images")

SENTIMENTS = ("positive", "neutral", "negative")

KEYWORDS = [
    "Scenery",
    "Nature",
    "History & Heritage",
    "Culture",
    "Architecture",
    "Food & Drink",
    "Shopping",
    "Activities",
    "Transportation",
    "Accessibility",
    "Facilities",
    "Service",
    "Price",
    "Crowds",
    "Cleanliness",
    "Safety",
    "Atmosphere",
]

ASPECT_INDEX = {aspect: index for index, aspect in enumerate(KEYWORDS)}

ASPECT_EMOJI = {
    "Scenery": "📸",
    "Nature": "🌳",
    "History & Heritage": "🏛",
    "Culture": "🎎",
    "Architecture": "🏯",
    "Food & Drink": "🍜",
    "Shopping": "🛍",
    "Activities": "🎡",
    "Transportation": "🚆",
    "Accessibility": "♿",
    "Facilities": "🏢",
    "Service": "🤝",
    "Price": "💸",
    "Crowds": "🔥",
    "Cleanliness": "✨",
    "Safety": "🛡",
    "Atmosphere": "🌅",
}

SENTIMENT_CONFIG = {
    "positive": ("😊", "Positive"),
    "neutral": ("😐", "Neutral"),
    "negative": ("🙁", "Negative"),
}

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


# =========================================================
# CSS
# =========================================================
APP_CSS = """
<style>
.stApp { background-color: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(135deg, #1E3A5F, #2563EB);
    padding: 42px 48px;
    border-radius: 28px;
    color: white;
    margin-bottom: 26px;
}
.hero-title { font-size: 46px; font-weight: 900; margin-bottom: 8px; }
.hero-sub { font-size: 18px; opacity: 0.92; }
.keyword-title { font-size: 24px; font-weight: 800; margin: 12px 0 8px; }

.selected-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 14px 16px;
    margin: 12px 0 22px;
    color: #1E3A5F;
}

.place-title { font-size: 23px; font-weight: 850; color: #1E3A5F; margin-bottom: 4px; }
.match-text { font-size: 16px; font-weight: 800; color: #2563EB; margin-bottom: 8px; }
.rating-text { font-size: 15px; font-weight: 750; color: #374151; margin-bottom: 10px; }

.intro-text {
    color: #374151;
    line-height: 1.6;
    margin-bottom: 12px;
    font-size: 15px;
    height: 72px;
    overflow-y: auto;
    padding-right: 4px;
}

.address-text { color: #6B7280; font-size: 14px; line-height: 1.5; margin-bottom: 14px; }
.fixed-img { width: 100%; height: 230px; object-fit: cover; border-radius: 16px; margin-bottom: 14px; }

.image-placeholder {
    width: 100%;
    height: 230px;
    border-radius: 16px;
    background-color: #E5E7EB;
    color: #64748B;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-bottom: 14px;
}

.highlight-item {
    background-color: #E0F2FE;
    color: #075985;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 650;
    display: inline-block;
    margin: 4px 4px 4px 0;
}

.section-label { font-weight: 850; margin: 10px 0 6px; color: #111827; }
.small-note { color: #64748B; font-size: 14px; }

.insight-box {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 16px;
    padding: 10px 15px;
    margin: 0;
    color: #334155;
    font-size: 14px;
    line-height: 1.55;
    box-sizing: border-box;
}
.insight-row { margin: 4px 0; }
.insight-label { font-weight: 750; }
.insight-topic { font-weight: 700; color: #334155; }
.insight-phrase { color: #64748B; font-size: 13px; }

.owi-box {
    border-radius: 16px;
    padding: 12px 15px;
    margin: 12px 0 14px;
    font-size: 14px;
    font-weight: 750;
}
.owi-high { background-color: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
.owi-medium { background-color: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.owi-low { background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; }

.tip-box {
    background-color: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 16px;
    padding: 14px 16px;
    margin: 10px 0 18px;
    color: #7C2D12;
    font-size: 14px;
    line-height: 1.7;
    height: 110px;
    box-sizing: border-box;
    overflow-y: auto;
    display: flex;
    align-items: center;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# =========================================================
# Generic helpers
# =========================================================
def safe_text(value: Any) -> str:
    """Return a stripped string, treating None/NaN as empty."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def clean_place_name(place_name: Any) -> str:
    """Remove a leading numeric prefix such as '1. '."""
    return re.sub(r"^\s*\d+\s*\.\s*", "", safe_text(place_name)).strip()


def normalize_place_name(place_name: Any) -> str:
    """Create a stable lowercase merge key for attraction names."""
    return re.sub(r"\s+", " ", clean_place_name(place_name)).strip().lower()


def escape(value: Any) -> str:
    return html.escape(safe_text(value))


def ensure_columns(df: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source_name}: {', '.join(missing)}")


# =========================================================
# Insight preprocessing
# =========================================================
def prepare_insight_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Attraction", "OWI", "Insight", "Sentiment", "Display Phrase", "Share %"]
    ensure_columns(raw_df, required, "Visitor Insights CSV")

    df = raw_df.copy()

    for column in ("Attraction", "Insight", "Display Phrase"):
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["Sentiment"] = df["Sentiment"].fillna("").astype(str).str.strip().str.lower()
    df["Share %"] = pd.to_numeric(df["Share %"], errors="coerce").fillna(0.0)
    df["OWI"] = pd.to_numeric(df["OWI"], errors="coerce")

    df = df[
        df["Sentiment"].isin(SENTIMENTS)
        & df["Attraction"].ne("")
        & df["Insight"].ne("")
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["merge_name", "OWI", "topic"])

    df["merge_name"] = df["Attraction"].map(normalize_place_name)

    top_rows = (
        df.sort_values(
            ["merge_name", "Sentiment", "Share %"],
            ascending=[True, True, False],
            kind="stable",
        )
        .drop_duplicates(["merge_name", "Sentiment"], keep="first")
    )

    topic_series = top_rows.groupby("merge_name", sort=False).apply(build_topic_dict)
    topic_df = topic_series.rename("topic").reset_index()

    owi_df = (
        df.dropna(subset=["OWI"])
        .drop_duplicates("merge_name", keep="first")[["merge_name", "OWI"]]
    )

    return topic_df.merge(owi_df, on="merge_name", how="left", validate="one_to_one")


def build_topic_dict(group: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {sentiment: None for sentiment in SENTIMENTS}

    for _, row in group.iterrows():
        sentiment = safe_text(row["Sentiment"]).lower()
        if sentiment not in result:
            continue

        result[sentiment] = {
            "topic": safe_text(row["Insight"]),
            "phrase": safe_text(row["Display Phrase"]),
            "share": float(row["Share %"]),
        }

    return result


# =========================================================
# Profile loading and vector preparation
# =========================================================
def validate_profile_data(df: pd.DataFrame) -> None:
    ensure_columns(
        df,
        ["location_name", "positive_top10", "negative_top10", "intro"],
        "profile CSV",
    )

    if df.empty:
        raise ValueError("The tourist profile data contains no rows.")
    if df["location_name"].isna().all():
        raise ValueError("All location_name values are missing.")


def parse_aspects(value: Any) -> list[tuple[str, float]]:
    text = safe_text(value)
    if not text:
        return []

    parsed: list[tuple[str, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(r"(.+?)\s*\(([\d.]+)\)", part)
        if not match:
            parsed.append((part, 1.0))
            continue

        aspect = match.group(1).strip()
        try:
            count = float(match.group(2))
        except ValueError:
            count = 1.0
        parsed.append((aspect, count))

    return parsed


def build_attraction_vector(positive_top10: Any) -> np.ndarray:
    vector = np.zeros(len(KEYWORDS), dtype=float)
    for aspect, count in parse_aspects(positive_top10):
        index = ASPECT_INDEX.get(aspect)
        if index is not None:
            vector[index] += count
    return vector


@st.cache_data
def load_data(profile_path: str, insight_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    profile_file = Path(profile_path)
    insight_file = Path(insight_path)

    if not profile_file.exists():
        raise FileNotFoundError(f"Profile data file not found: {profile_file}")
    if not insight_file.exists():
        raise FileNotFoundError(f"Visitor Insights CSV file not found: {insight_file}")

    profile_df = pd.read_csv(profile_file)
    insight_df = prepare_insight_data(pd.read_csv(insight_file))

    validate_profile_data(profile_df)

    profile_df = profile_df.copy()
    profile_df["merge_name"] = profile_df["location_name"].map(normalize_place_name)
    profile_df = profile_df.drop(columns=["OWI", "topic"], errors="ignore")

    merged = profile_df.merge(
        insight_df[["merge_name", "OWI", "topic"]],
        on="merge_name",
        how="left",
        validate="one_to_one",
    ).drop(columns="merge_name")

    attraction_matrix = np.vstack(merged["positive_top10"].map(build_attraction_vector))
    return merged, attraction_matrix


def build_user_vector(selected_keywords: list[str]) -> np.ndarray:
    vector = np.zeros(len(KEYWORDS), dtype=float)
    for keyword in selected_keywords:
        index = ASPECT_INDEX.get(keyword)
        if index is not None:
            vector[index] = 1.0
    return vector


def calculate_cosine_scores(selected_keywords: list[str], attraction_matrix: np.ndarray) -> np.ndarray:
    user_vector = build_user_vector(selected_keywords)
    user_norm = np.linalg.norm(user_vector)

    if user_norm == 0:
        return np.zeros(attraction_matrix.shape[0], dtype=float)

    item_norms = np.linalg.norm(attraction_matrix, axis=1)
    denominators = item_norms * user_norm
    dot_products = attraction_matrix @ user_vector

    return np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominators != 0,
    )


# =========================================================
# Image helpers
# =========================================================
@st.cache_data
def image_to_data_uri(path: str) -> str:
    image_path = Path(path)
    mime_extension = "jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else image_path.suffix[1:].lower()
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime_extension};base64,{encoded}"


def find_image_path(place_name: str) -> Path | None:
    clean_name = clean_place_name(place_name)
    for extension in IMAGE_EXTENSIONS:
        for candidate_extension in (extension, extension.upper()):
            candidate = IMAGE_DIR / f"{clean_name}.{candidate_extension}"
            if candidate.exists():
                return candidate
    return None


def render_image(place_name: str) -> None:
    image_path = find_image_path(place_name)
    if image_path is None:
        st.markdown(
            '<div class="image-placeholder">Image not available</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<img src="{image_to_data_uri(str(image_path))}" class="fixed-img">',
        unsafe_allow_html=True,
    )


# =========================================================
# HTML builders
# =========================================================
def highlight_badges(items: list[tuple[str, float]], limit: int = 5) -> str:
    if not items:
        return '<span class="small-note">No highlight data available</span>'

    badges = []
    for aspect, _ in items[:limit]:
        badges.append(
            f'<span class="highlight-item">{ASPECT_EMOJI.get(aspect, "✨")} {escape(aspect)}</span>'
        )
    return "".join(badges)


def parse_topic_dict(value: Any) -> dict[str, dict[str, Any] | None]:
    empty = {sentiment: None for sentiment in SENTIMENTS}

    if isinstance(value, dict):
        parsed = value
    else:
        text = safe_text(value)
        if not text:
            return empty

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return empty

    if not isinstance(parsed, dict):
        return empty

    result = empty.copy()
    for sentiment in SENTIMENTS:
        item = parsed.get(sentiment)
        if not isinstance(item, dict):
            continue

        topic = safe_text(item.get("topic"))
        if not topic:
            continue

        result[sentiment] = {
            "topic": topic,
            "phrase": safe_text(item.get("phrase")),
            "share": item.get("share"),
        }

    return result


def visitor_insights_html(value: Any) -> str:
    rows = []
    for sentiment, item in parse_topic_dict(value).items():
        if item is None:
            continue

        emoji, label = SENTIMENT_CONFIG[sentiment]
        phrase = escape(item.get("phrase"))
        content = f'<span class="insight-topic">{escape(item.get("topic"))}</span>'
        if phrase:
            content += f' — <span class="insight-phrase">{phrase}</span>'

        rows.append(
            '<div class="insight-row">'
            f'<span class="insight-label">{emoji} {label}:</span> {content}'
            "</div>"
        )

    return f'<div class="insight-box">{"".join(rows)}</div>' if rows else ""


def owi_html(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(score):
        return ""

    if score > 0.6:
        level, emoji, css_class = "High", "🚨", "owi-high"
    elif score > 0.4:
        level, emoji, css_class = "Medium", "⚠️", "owi-medium"
    else:
        level, emoji, css_class = "Low", "✅", "owi-low"

    return (
        f'<div class="owi-box {css_class}">'
        f'OWI (Overcrowding Warning Index): {level} {emoji} ({score:.2f})'
        "</div>"
    )


def section_label(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def rank_badge(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")


# =========================================================
# Rendering
# =========================================================
def render_keyword_selector() -> list[str]:
    st.markdown('<div class="keyword-title">Choose Your Travel Style</div>', unsafe_allow_html=True)
    st.caption("Select one or more preferences. Recommendations update automatically.")

    columns = st.columns(5)
    for index, aspect in enumerate(KEYWORDS):
        selected = aspect in st.session_state.selected_keywords
        prefix = "✓ " if selected else ""
        label = f"{prefix}{ASPECT_EMOJI.get(aspect, '✨')} {aspect}"

        with columns[index % 5]:
            if st.button(label, key=f"kw_{aspect}", use_container_width=True):
                toggle_keyword(aspect)
                st.rerun()

    selected_keywords = st.session_state.selected_keywords
    if selected_keywords:
        st.markdown(
            f'<div class="selected-box"><b>Selected Preferences:</b> {escape(", ".join(selected_keywords))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Select your travel preferences above to start exploring recommended attractions.")

    return selected_keywords


def render_card(row: pd.Series, rank: int) -> None:
    place_name = clean_place_name(row.get("location_name"))
    intro = safe_text(row.get("intro"))
    address = safe_text(row.get("english_address"))
    highlights = parse_aspects(row.get("positive_top10"))

    with st.container(border=True):
        render_image(place_name)
        # Location name
        st.markdown(
            f'<div class="place-title">{rank_badge(rank)} {escape(place_name)}</div>',
            unsafe_allow_html=True,
        )

        # Address
        if address:
            st.markdown(
                f'<div class="address-text">📍 {escape(address)}</div>',
                unsafe_allow_html=True,
            )

        # Match
        st.markdown(
            f'<div class="match-text">⭐ {int(row["match_percent"])}% Match</div>',
            unsafe_allow_html=True,
        )

        # Rating
        rating = row.get("avg_rating")
        if pd.notna(rating):
            st.markdown(
                f'<div class="rating-text">⭐ Visitor Rating: {float(rating):.1f}/5.0</div>',
                unsafe_allow_html=True,
            )

        # Description
        if intro:
            st.markdown(
                f'<div class="intro-text">{escape(intro)}</div>',
                unsafe_allow_html=True,
            )
        section_label("✨ Highlights")
        st.markdown(
            highlight_badges(highlights),
            unsafe_allow_html=True,
        )

        section_label("💬 Visitor Insights")
        insight_markup = visitor_insights_html(row.get("topic"))

        st.markdown(
            insight_markup
            or '<div class="insight-box"><span class="small-note">No visitor insight data available</span></div>',
            unsafe_allow_html=True,
        )

        overcrowding_markup = owi_html(row.get("OWI"))
        if overcrowding_markup:
            st.markdown(overcrowding_markup, unsafe_allow_html=True)

        section_label("💡 Good to Know")
        note = safe_text(row.get("good_to_know"))
        note_markup = escape(note) if note else '<span class="small-note">No additional travel notes available</span>'
        st.markdown(f'<div class="tip-box">{note_markup}</div>', unsafe_allow_html=True)


def render_recommendations(df: pd.DataFrame, attraction_matrix: np.ndarray, selected: list[str]) -> None:
    result = df.copy()
    result["similarity"] = calculate_cosine_scores(selected, attraction_matrix)
    result = result.nlargest(TOP_N, "similarity").copy()
    result["match_percent"] = (result["similarity"].clip(0, 1) * 100).round().astype(int)

    st.markdown("## ⭐ Recommended Attractions")
    columns = st.columns(2)

    for rank, (_, row) in enumerate(result.iterrows(), start=1):
        with columns[(rank - 1) % 2]:
            render_card(row, rank)


def render_initial_explanation() -> None:
    st.markdown("## How it works")
    st.write(
        "This system builds positive aspect profiles for each tourist attraction using ABSA results "
        "extracted from tourist reviews. User preferences and tourist attractions are represented as "
        "aspect vectors, and recommendations are ranked using cosine similarity. The cosine similarity "
        "score is multiplied by 100 and displayed as Match (%)."
    )


def toggle_keyword(value: str) -> None:
    selected = st.session_state.selected_keywords
    if value in selected:
        selected.remove(value)
    else:
        selected.append(value)


# =========================================================
# Main
# =========================================================
def main() -> None:
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = []

    try:
        df, attraction_matrix = load_data(str(PROFILE_CSV_PATH), str(INSIGHT_CSV_PATH))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Check that both CSV files exist in the data folder and that their filenames match the settings.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error("An unexpected error occurred while loading data.")
        st.exception(exc)
        st.stop()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🌏 SmartTrip AI</div>
            <div class="hero-sub">Discover Must-Visit Attractions in South Korea.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = render_keyword_selector()
    if selected:
        render_recommendations(df, attraction_matrix, selected)
    else:
        render_initial_explanation()


if __name__ == "__main__":
    main()