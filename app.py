import os
import re
import base64
import html
import ast
import json

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Streamlit page settings
# =========================================================
st.set_page_config(
    page_title="SmartTrip AI",
    page_icon="🌏",
    layout="wide",
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #1E3A5F, #2563EB);
        padding: 42px 48px;
        border-radius: 28px;
        color: white;
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero-sub {
        font-size: 18px;
        opacity: 0.92;
    }

    .keyword-title {
        font-size: 24px;
        font-weight: 800;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .selected-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 12px;
        margin-bottom: 22px;
        color: #1E3A5F;
    }

    .place-title {
        font-size: 23px;
        font-weight: 850;
        color: #1E3A5F;
        margin-bottom: 4px;
    }

    .match-text {
        font-size: 16px;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 8px;
    }

    .rating-text {
        font-size: 15px;
        font-weight: 750;
        color: #374151;
        margin-bottom: 10px;
    }

    .intro-text {
        color: #374151;
        line-height: 1.6;
        margin-bottom: 12px;
        font-size: 15px;
        height: 72px;
        overflow-y: auto;
        padding-right: 4px;
    }

    .address-text {
        color: #6B7280;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 14px;
    }

    .fixed-img {
        width: 100%;
        height: 230px;
        object-fit: cover;
        border-radius: 16px;
        margin-bottom: 14px;
    }

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

    .section-label {
        font-weight: 850;
        margin-top: 16px;
        margin-bottom: 7px;
        color: #111827;
    }

    .tip-box {
        background-color: #FFF7ED;
        border: 1px solid #FED7AA;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 18px;
        color: #7C2D12;
        font-size: 14px;
        line-height: 1.7;
        height: 110px;
        box-sizing: border-box;
        overflow-y: auto;
        display: flex;
        align-items: center;
    }

    .insight-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 16px;
        padding: 10px 15px;
        margin-top: 0;
        margin-bottom: 0;
        color: #334155;
        font-size: 14px;
        line-height: 1.55;
        min-height: 92px;
        max-height: 180px;
        box-sizing: border-box;
        overflow-y: auto;
    }

    .insight-row {
        margin: 4px 0;
    }

    .insight-label {
        font-weight: 750;
    }

    .insight-topic {
        font-weight: 700;
        color: #334155;
    }

    .insight-phrase {
        color: #64748B;
        font-size: 13px;
    }INSIGHT_PATH

    .owi-box {
        border-radius: 16px;
        padding: 12px 15px;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 14px;
        font-weight: 750;
    }

    .owi-high {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }

    .owi-medium {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }

    .owi-low {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }

    .small-note {
        color: #64748B;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Settings
# =========================================================
TOP_N = 6

PROFILE_CSV_PATH = "data/tourist_profile_data_with_tips.csv"

INSIGHT_PATH = (
    "data/Visitor_Insights_Keywords_OWI_v4.csv"
)

INSIGHT_SHEET_NAME = "Visitor Insights (long)"


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


ASPECT_INDEX = {
    aspect: index
    for index, aspect in enumerate(KEYWORDS)
}


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


# =========================================================
# General helper functions
# =========================================================
def safe_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def clean_place_name(place_name):
    """
    Remove a leading location number.

    Example:
    '1. Gyeongbokgung Palace'
    -> 'Gyeongbokgung Palace'
    """
    return re.sub(
        r"^\s*\d+\s*\.\s*",
        "",
        safe_text(place_name),
    ).strip()


def normalize_place_name(place_name):
    """
    Normalize attraction names for merging.

    - Remove leading number
    - Remove repeated spaces
    - Convert to lowercase
    """
    cleaned = clean_place_name(place_name)

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned.lower()


# =========================================================
# Load and preprocess Visitor Insights
# =========================================================
def prepare_insight_data(insight_df):
    required_columns = [
        "Attraction",
        "OWI",
        "Insight",
        "Sentiment",
        "Display Phrase",
        "Share %",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in insight_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in Visitor Insights Excel file: "
            + ", ".join(missing_columns)
        )

    insight_df = insight_df.copy()

    # Clean text columns
    insight_df["Attraction"] = (
        insight_df["Attraction"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    insight_df["Insight"] = (
        insight_df["Insight"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    insight_df["Display Phrase"] = (
        insight_df["Display Phrase"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    insight_df["Sentiment"] = (
        insight_df["Sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert numeric columns
    insight_df["Share %"] = pd.to_numeric(
        insight_df["Share %"],
        errors="coerce",
    ).fillna(0.0)

    insight_df["OWI"] = pd.to_numeric(
        insight_df["OWI"],
        errors="coerce",
    )

    # Keep P/N/N rows only
    insight_df = insight_df[
        insight_df["Sentiment"].isin(
            [
                "positive",
                "neutral",
                "negative",
            ]
        )
    ].copy()

    # Remove rows without an attraction or Insight
    insight_df = insight_df[
        insight_df["Attraction"].ne("")
        & insight_df["Insight"].ne("")
    ].copy()

    insight_df["merge_name"] = (
        insight_df["Attraction"]
        .apply(normalize_place_name)
    )

    # -----------------------------------------------------
    # Select one row per attraction and sentiment.
    #
    # If two or more rows have the same attraction and
    # sentiment, select the row with the highest Share %.
    # -----------------------------------------------------
    insight_top1 = (
        insight_df
        .sort_values(
            by=[
                "merge_name",
                "Sentiment",
                "Share %",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "merge_name",
                "Sentiment",
            ],
            keep="first",
        )
        .copy()
    )

    # -----------------------------------------------------
    # Convert long-form PNN rows into one dictionary
    # per attraction.
    # -----------------------------------------------------
    topic_records = []

    for merge_name, group in insight_top1.groupby(
        "merge_name",
        sort=False,
    ):
        topic_dict = {
            "positive": None,
            "neutral": None,
            "negative": None,
        }

        attraction_name = safe_text(
            group["Attraction"].iloc[0]
        )

        for _, row in group.iterrows():
            sentiment = safe_text(
                row["Sentiment"]
            ).lower()

            if sentiment not in topic_dict:
                continue

            topic_dict[sentiment] = {
                "topic": safe_text(
                    row["Insight"]
                ),
                "phrase": safe_text(
                    row["Display Phrase"]
                ),
                "share": float(
                    row["Share %"]
                ),
            }

        topic_records.append(
            {
                "merge_name": merge_name,
                "Attraction": attraction_name,
                "topic": topic_dict,
            }
        )

    topic_df = pd.DataFrame(topic_records)

    # -----------------------------------------------------
    # One OWI value per attraction
    # -----------------------------------------------------
    owi_df = (
        insight_df[
            [
                "merge_name",
                "OWI",
            ]
        ]
        .dropna(
            subset=["OWI"]
        )
        .drop_duplicates(
            subset=["merge_name"],
            keep="first",
        )
        .copy()
    )

    if topic_df.empty:
        return pd.DataFrame(
            columns=[
                "merge_name",
                "Attraction",
                "topic",
                "OWI",
            ]
        )

    prepared_df = topic_df.merge(
        owi_df,
        on="merge_name",
        how="left",
        validate="one_to_one",
    )

    return prepared_df


# =========================================================
# Load data
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(PROFILE_CSV_PATH):
        raise FileNotFoundError(
            f"Profile data file not found: "
            f"{PROFILE_CSV_PATH}"
        )

    if not os.path.exists(INSIGHT_PATH):
        raise FileNotFoundError(
            f"Visitor Insights Excel file not found: "
            f"{INSIGHT_PATH}"
        )

    profile_df = pd.read_csv(
        PROFILE_CSV_PATH
    )

    raw_insight_df = pd.read_csv(
        INSIGHT_PATH,
        sheet_name=INSIGHT_SHEET_NAME,
    )

    prepared_insight_df = prepare_insight_data(
        raw_insight_df
    )

    if "location_name" not in profile_df.columns:
        raise ValueError(
            "The profile CSV must contain "
            "a location_name column."
        )

    # Create merge key after removing location numbers
    profile_df["merge_name"] = (
        profile_df["location_name"]
        .apply(normalize_place_name)
    )

    # Remove old OWI/topic columns before merging
    profile_df = profile_df.drop(
        columns=[
            "OWI",
            "topic",
        ],
        errors="ignore",
    )

    merged_df = profile_df.merge(
        prepared_insight_df[
            [
                "merge_name",
                "OWI",
                "topic",
            ]
        ],
        on="merge_name",
        how="left",
        validate="one_to_one",
    )

    merged_df = merged_df.drop(
        columns=["merge_name"],
        errors="ignore",
    )

    return merged_df


try:
    df = load_data()

except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Check that the CSV and XLSX files are inside "
        "the data folder and that their filenames match "
        "the paths in the Settings section."
    )
    st.stop()

except ValueError as exc:
    st.error(str(exc))
    st.stop()

except Exception as exc:
    st.error(
        "An unexpected error occurred while loading data."
    )
    st.exception(exc)
    st.stop()


# =========================================================
# Image helper functions
# =========================================================
def get_local_image_path(place_name):
    clean_name = clean_place_name(
        place_name
    )

    extensions = [
        "jpg",
        "jpeg",
        "JPG",
        "JPEG",
        "png",
        "PNG",
        "webp",
        "WEBP",
    ]

    for extension in extensions:
        path = os.path.join(
            "images",
            f"{clean_name}.{extension}",
        )

        if os.path.exists(path):
            return path

    return None


def image_to_base64(path):
    with open(path, "rb") as file:
        data = file.read()

    extension = (
        os.path.splitext(path)[1]
        .lower()
        .replace(".", "")
    )

    if extension == "jpg":
        extension = "jpeg"

    encoded = base64.b64encode(
        data
    ).decode()

    return (
        f"data:image/{extension};base64,"
        f"{encoded}"
    )


def show_fixed_image(image_path):
    if image_path:
        image_source = image_to_base64(
            image_path
        )

        st.markdown(
            (
                f'<img src="{image_source}" '
                f'class="fixed-img">'
            ),
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="image-placeholder">
                Image not available
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# Aspect parsing and recommendation functions
# =========================================================
def parse_aspects(text):
    if text is None:
        return []

    try:
        if pd.isna(text):
            return []
    except (TypeError, ValueError):
        pass

    items = []

    for part in str(text).split(","):
        part = part.strip()

        if not part:
            continue

        match = re.match(
            r"(.+?)\(([\d.]+)\)",
            part,
        )

        if match:
            aspect = match.group(1).strip()

            try:
                count = float(
                    match.group(2)
                )
            except ValueError:
                count = 1.0

            items.append(
                (
                    aspect,
                    count,
                )
            )

        else:
            items.append(
                (
                    part,
                    1.0,
                )
            )

    return items


def validate_profile_data(profile_df):
    required_columns = [
        "location_name",
        "positive_top10",
        "negative_top10",
        "intro",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in profile_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required profile columns: "
            + ", ".join(missing_columns)
        )

    if profile_df.empty:
        raise ValueError(
            "The tourist profile data contains no rows."
        )

    if profile_df["location_name"].isna().all():
        raise ValueError(
            "All location_name values are missing."
        )


def build_user_vector(selected_keywords):
    vector = np.zeros(
        len(KEYWORDS),
        dtype=float,
    )

    for aspect in selected_keywords:
        index = ASPECT_INDEX.get(
            aspect
        )

        if index is not None:
            vector[index] = 1.0

    return vector


def build_attraction_vector(positive_top10):
    """
    Build a positive-aspect frequency vector.

    Example:
    Nature(120), Scenery(80)

    becomes:
    Nature = 120
    Scenery = 80
    Other aspects = 0
    """
    vector = np.zeros(
        len(KEYWORDS),
        dtype=float,
    )

    for aspect, count in parse_aspects(
        positive_top10
    ):
        index = ASPECT_INDEX.get(
            aspect
        )

        if index is not None:
            vector[index] += float(
                count
            )

    return vector


def cosine_similarity(vector_a, vector_b):
    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / (
            norm_a
            * norm_b
        )
    )


def calculate_aspect_cosine_scores(
    selected_keywords,
    profile_df,
):
    user_vector = build_user_vector(
        selected_keywords
    )

    scores = []

    for _, row in profile_df.iterrows():
        attraction_vector = (
            build_attraction_vector(
                row.get(
                    "positive_top10",
                    "",
                )
            )
        )

        score = cosine_similarity(
            user_vector,
            attraction_vector,
        )

        scores.append(
            score
        )

    return np.array(
        scores,
        dtype=float,
    )


# =========================================================
# Highlight display
# =========================================================
def highlight_badges(items, limit=5):
    if not items:
        return (
            "<span class='small-note'>"
            "No highlight data available"
            "</span>"
        )

    html_code = ""

    for aspect, _ in items[:limit]:
        aspect = safe_text(
            aspect
        )

        emoji = ASPECT_EMOJI.get(
            aspect,
            "✨",
        )

        html_code += (
            "<span class='highlight-item'>"
            f"{emoji} {html.escape(aspect)}"
            "</span>"
        )

    return html_code


# =========================================================
# Visitor Insights parsing and display
# =========================================================
def parse_topic_dict(value):
    empty_result = {
        "positive": None,
        "neutral": None,
        "negative": None,
    }

    if isinstance(value, dict):
        parsed = value

    elif value is None:
        return empty_result

    else:
        try:
            if pd.isna(value):
                return empty_result
        except (TypeError, ValueError):
            pass

        raw_value = str(value).strip()

        if not raw_value:
            return empty_result

        try:
            parsed = ast.literal_eval(
                raw_value
            )

        except (
            ValueError,
            SyntaxError,
        ):
            try:
                parsed = json.loads(
                    raw_value
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                return empty_result

    if not isinstance(parsed, dict):
        return empty_result

    result = empty_result.copy()

    for sentiment in [
        "positive",
        "neutral",
        "negative",
    ]:
        item = parsed.get(
            sentiment
        )

        if not isinstance(item, dict):
            continue

        topic = safe_text(
            item.get("topic")
        )

        phrase = safe_text(
            item.get("phrase")
        )

        share = item.get(
            "share"
        )

        if not topic:
            continue

        result[sentiment] = {
            "topic": topic,
            "phrase": phrase,
            "share": share,
        }

    return result


def visitor_insights_html(topic_value):
    topics = parse_topic_dict(
        topic_value
    )

    sentiment_config = {
        "positive": (
            "😊",
            "Positive",
        ),
        "neutral": (
            "😐",
            "Neutral",
        ),
        "negative": (
            "🙁",
            "Negative",
        ),
    }

    rows = []

    for sentiment in [
        "positive",
        "neutral",
        "negative",
    ]:
        item = topics.get(
            sentiment
        )

        if item is None:
            continue

        emoji, label = sentiment_config[
            sentiment
        ]

        topic_text = html.escape(
            safe_text(
                item.get("topic")
            )
        )

        phrase_text = html.escape(
            safe_text(
                item.get("phrase")
            )
        )

        if phrase_text:
            insight_content = (
                f"<span class='insight-topic'>"
                f"{topic_text}"
                f"</span>"
                f" — "
                f"<span class='insight-phrase'>"
                f"{phrase_text}"
                f"</span>"
            )

        else:
            insight_content = (
                f"<span class='insight-topic'>"
                f"{topic_text}"
                f"</span>"
            )

        rows.append(
            (
                "<div class='insight-row'>"
                f"<span class='insight-label'>"
                f"{emoji} {label}:"
                f"</span> "
                f"{insight_content}"
                "</div>"
            )
        )

    if not rows:
        return ""

    return (
        "<div class='insight-box'>"
        + "".join(rows)
        + "</div>"
    )


# =========================================================
# OWI display
# =========================================================
def get_owi_display(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        score = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    # High: score > 0.6
    # Medium: 0.4 < score <= 0.6
    # Low: score <= 0.4
    if score > 0.6:
        return (
            "High",
            "🚨",
            "owi-high",
            score,
        )

    if score > 0.4:
        return (
            "Medium",
            "⚠️",
            "owi-medium",
            score,
        )

    return (
        "Low",
        "✅",
        "owi-low",
        score,
    )


def owi_html(value):
    display = get_owi_display(
        value
    )

    if display is None:
        return ""

    level, emoji, css_class, score = display

    return (
        f"<div class='owi-box {css_class}'>"
        "OWI (Overcrowding Warning Index): "
        f"{level} {emoji} ({score:.2f})"
        "</div>"
    )


def get_good_to_know(row):
    return safe_text(
        row.get(
            "good_to_know",
            "",
        )
    )


# =========================================================
# Validate profile data
# =========================================================
try:
    validate_profile_data(
        df
    )

except ValueError as exc:
    st.error(
        str(exc)
    )
    st.stop()


# =========================================================
# Session state
# =========================================================
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []


def toggle_keyword(value):
    if value in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(
            value
        )

    else:
        st.session_state.selected_keywords.append(
            value
        )


# =========================================================
# Hero
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🌏 SmartTrip AI
        </div>

        <div class="hero-sub">
            Discover Must-Visit Attractions in South Korea.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Keyword selector
# =========================================================
st.markdown(
    (
        '<div class="keyword-title">'
        "Choose Your Travel Style"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.caption(
    "Select one or more preferences. "
    "Recommendations update automatically."
)

keyword_columns = st.columns(
    5
)

for index, aspect in enumerate(KEYWORDS):
    emoji = ASPECT_EMOJI.get(
        aspect,
        "✨",
    )

    selected = (
        aspect
        in st.session_state.selected_keywords
    )

    if selected:
        button_label = (
            f"✓ {emoji} {aspect}"
        )

    else:
        button_label = (
            f"{emoji} {aspect}"
        )

    with keyword_columns[index % 5]:
        if st.button(
            button_label,
            key=f"kw_{aspect}",
            use_container_width=True,
        ):
            toggle_keyword(
                aspect
            )

            st.rerun()


if st.session_state.selected_keywords:
    selected_text = ", ".join(
        st.session_state.selected_keywords
    )

    st.markdown(
        f"""
        <div class="selected-box">
            <b>Selected Preferences:</b>
            {html.escape(selected_text)}
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info(
        "Select your travel preferences above "
        "to start exploring recommended attractions."
    )


# =========================================================
# Recommendation results
# =========================================================
if st.session_state.selected_keywords:
    similarities = calculate_aspect_cosine_scores(
        st.session_state.selected_keywords,
        df,
    )

    result = df.copy()

    result["similarity"] = similarities

    result = (
        result
        .sort_values(
            by="similarity",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    result["match_percent"] = (
        result["similarity"]
        .clip(
            lower=0,
            upper=1,
        )
        .mul(100)
        .round()
        .astype(int)
    )

    st.markdown(
        "## ⭐ Recommended Attractions"
    )

    card_columns = st.columns(
        2
    )

    for card_index, (_, row) in enumerate(
        result.iterrows()
    ):
        place_name = clean_place_name(
            row["location_name"]
        )

        positive_items = parse_aspects(
            row.get(
                "positive_top10",
                "",
            )
        )

        image_path = get_local_image_path(
            place_name
        )

        intro = safe_text(
            row.get(
                "intro",
                "",
            )
        )

        address = safe_text(
            row.get(
                "english_address",
                "",
            )
        )

        good_to_know = get_good_to_know(
            row
        )

        rank = card_index + 1

        if rank == 1:
            medal = "🥇"

        elif rank == 2:
            medal = "🥈"

        elif rank == 3:
            medal = "🥉"

        else:
            medal = f"{rank}."

        with card_columns[card_index % 2]:
            with st.container(
                border=True
            ):
                show_fixed_image(
                    image_path
                )

                st.markdown(
                    f"""
                    <div class="place-title">
                        {medal} {html.escape(place_name)}
                    </div>

                    <div class="match-text">
                        ⭐ {row["match_percent"]}% Match
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if (
                    "avg_rating" in row.index
                    and pd.notna(
                        row["avg_rating"]
                    )
                ):
                    st.markdown(
                        f"""
                        <div class="rating-text">
                            ⭐ Visitor Rating:
                            {float(row["avg_rating"]):.1f}/5.0
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if intro:
                    st.markdown(
                        f"""
                        <div class="intro-text">
                            {html.escape(intro)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if address:
                    st.markdown(
                        f"""
                        <div class="address-text">
                            📍 {html.escape(address)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------
                # Visitor Insights
                # P/N/N maximum one row each
                # -----------------------------------------
                insight_html = visitor_insights_html(
                    row.get(
                        "topic",
                        "",
                    )
                )

                st.markdown(
                    (
                        '<div class="section-label">'
                        "💬 Visitor Insights"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                if insight_html:
                    st.markdown(
                        insight_html,
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown(
                        """
                        <div class="insight-box">
                            <span class="small-note">
                                No visitor insight data available
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------
                # OWI
                # -----------------------------------------
                overcrowding_html = owi_html(
                    row.get(
                        "OWI",
                        np.nan,
                    )
                )

                if overcrowding_html:
                    st.markdown(
                        overcrowding_html,
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------
                # Highlights
                # -----------------------------------------
                st.markdown(
                    (
                        '<div class="section-label">'
                        "✨ Highlights"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    highlight_badges(
                        positive_items,
                        limit=5,
                    ),
                    unsafe_allow_html=True,
                )

                # -----------------------------------------
                # Good to Know
                # -----------------------------------------
                st.markdown(
                    (
                        '<div class="section-label">'
                        "💡 Good to Know"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                if good_to_know:
                    tip_content = html.escape(
                        good_to_know
                    )

                else:
                    tip_content = (
                        '<span class="small-note">'
                        "No additional travel notes available"
                        "</span>"
                    )

                st.markdown(
                    f"""
                    <div class="tip-box">
                        {tip_content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =========================================================
# Initial explanation
# =========================================================
else:
    st.markdown(
        "## How it works"
    )

    st.write(
        "This system builds positive aspect profiles for each "
        "tourist attraction using ABSA results extracted from "
        "tourist reviews. User preferences and tourist attractions "
        "are represented as aspect vectors, and recommendations are "
        "ranked using cosine similarity. The cosine similarity score "
        "is multiplied by 100 and displayed as Match (%)."
    )