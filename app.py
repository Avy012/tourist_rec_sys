import os
import re
import base64
import html

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SmartTrip AI",
    page_icon="🌏",
    layout="wide"
)


# =========================
# CSS
# =========================
st.markdown("""
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
    background-color: #e5e7eb;
    color: #64748b;
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
    min-height: 110px;
    display: flex;
    align-items: center;
}

.small-note {
    color: #64748b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Settings
# =========================
TOP_N = 6

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


# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/tourist_profile_data.csv")


df = load_data()


# =========================
# Helper functions
# =========================
def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_place_name(place_name):
    return re.sub(r"^\d+\.\s*", "", str(place_name)).strip()


def get_local_image_path(place_name):
    clean_name = clean_place_name(place_name)

    for ext in ["jpg", "jpeg", "JPG", "png", "webp"]:
        path = os.path.join("images", f"{clean_name}.{ext}")
        if os.path.exists(path):
            return path

    return None


def image_to_base64(path):
    with open(path, "rb") as file:
        data = file.read()

    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"

    encoded = base64.b64encode(data).decode()
    return f"data:image/{ext};base64,{encoded}"


def show_fixed_image(img_path):
    if img_path:
        img_src = image_to_base64(img_path)
        st.markdown(
            f'<img src="{img_src}" class="fixed-img">',
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


def parse_aspects(text):
    if pd.isna(text):
        return []

    items = []

    for part in str(text).split(","):
        part = part.strip()

        if not part:
            continue

        match = re.match(r"(.+?)\((\d+)\)", part)

        if match:
            aspect = match.group(1).strip()
            count = int(match.group(2))
            items.append((aspect, count))
        else:
            items.append((part, 1))

    return items


def validate_profile_data(df_profile):
    required_columns = [
        "location_name",
        "positive_top10",
        "negative_top10",
        "intro",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_profile.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if df_profile.empty:
        raise ValueError("The tourist profile data contains no rows.")

    if df_profile["location_name"].isna().all():
        raise ValueError("All location_name values are missing.")


def build_user_vector(selected_keywords):
    vector = np.zeros(len(KEYWORDS), dtype=float)

    for aspect in selected_keywords:
        index = ASPECT_INDEX.get(aspect)
        if index is not None:
            vector[index] = 1.0

    return vector


def build_attraction_vector(positive_top10):
    """
    Build a positive-aspect frequency vector.

    Example:
    Nature(120), Scenery(80) ->
    [Scenery=80, Nature=120, other aspects=0]

    Cosine similarity is scale-invariant, so attractions with the same
    aspect proportions have the same direction regardless of total counts.
    """
    vector = np.zeros(len(KEYWORDS), dtype=float)

    for aspect, count in parse_aspects(positive_top10):
        index = ASPECT_INDEX.get(aspect)

        if index is not None:
            vector[index] += float(count)

    return vector


def cosine_similarity(vector_a, vector_b):
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def calculate_aspect_cosine_scores(selected_keywords, df_profile):
    user_vector = build_user_vector(selected_keywords)

    scores = []

    for _, row in df_profile.iterrows():
        attraction_vector = build_attraction_vector(
            row.get("positive_top10", "")
        )

        score = cosine_similarity(
            user_vector,
            attraction_vector,
        )
        scores.append(score)

    return np.array(scores, dtype=float)


def highlight_badges(items, limit=5):
    if not items:
        return (
            "<span class='small-note'>"
            "No highlight data available"
            "</span>"
        )

    html_code = ""

    for aspect, _ in items[:limit]:
        aspect = safe_text(aspect)
        emoji = ASPECT_EMOJI.get(aspect, "✨")

        html_code += (
            "<span class='highlight-item'>"
            f"{emoji} {html.escape(aspect)}"
            "</span>"
        )

    return html_code


def get_good_to_know(row):
    return safe_text(row.get("good_to_know", ""))


# =========================
# Validate data
# =========================
try:
    validate_profile_data(df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# =========================
# Session state
# =========================
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []


def toggle_keyword(value):
    if value in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(value)
    else:
        st.session_state.selected_keywords.append(value)


# =========================
# Hero
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">🌏 SmartTrip AI</div>
    <div class="hero-sub">
        Discover Must-Visit Attractions in South Korea.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Keyword selector
# =========================
st.markdown(
    '<div class="keyword-title">Choose Your Travel Style</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Select one or more preferences. "
    "Recommendations update automatically."
)

cols = st.columns(5)

for i, aspect in enumerate(KEYWORDS):
    emoji = ASPECT_EMOJI.get(aspect, "✨")
    selected = aspect in st.session_state.selected_keywords

    btn_label = (
        f"✓ {emoji} {aspect}"
        if selected
        else f"{emoji} {aspect}"
    )

    with cols[i % 5]:
        if st.button(
            btn_label,
            key=f"kw_{aspect}",
            use_container_width=True,
        ):
            toggle_keyword(aspect)
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
        "Select your travel preferences above to start "
        "exploring recommended attractions."
    )


# =========================
# Recommendation results
# =========================
if st.session_state.selected_keywords:
    similarities = calculate_aspect_cosine_scores(
        st.session_state.selected_keywords,
        df,
    )

    result = df.copy()
    result["similarity"] = similarities

    result = (
        result
        .sort_values("similarity", ascending=False)
        .head(TOP_N)
        .copy()
    )

    # Directly convert cosine similarity (0-1) to Match percentage.
    result["match_percent"] = (
        result["similarity"]
        .clip(0, 1)
        .mul(100)
        .round()
        .astype(int)
    )

    st.markdown("## ⭐ Recommended Attractions")

    card_cols = st.columns(2)

    for idx, (_, row) in enumerate(result.iterrows()):
        place_name = clean_place_name(row["location_name"])
        pos_items = parse_aspects(
            row.get("positive_top10", "")
        )
        img_path = get_local_image_path(place_name)

        intro = safe_text(row.get("intro", ""))
        address = safe_text(
            row.get("english_address", "")
        )
        good_to_know = get_good_to_know(row)

        rank = idx + 1

        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = f"{rank}."

        with card_cols[idx % 2]:
            with st.container(border=True):
                show_fixed_image(img_path)

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
                    and pd.notna(row["avg_rating"])
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

                st.markdown(
                    '<div class="section-label">'
                    '✨ Highlights'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    highlight_badges(
                        pos_items,
                        limit=5,
                    ),
                    unsafe_allow_html=True,
                )

                if good_to_know:
                    st.markdown(
                        '<div class="section-label">'
                        '💡 Good to Know'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div class="tip-box">
                            {html.escape(good_to_know)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

else:
    st.markdown("## How it works")
    st.write(
        "This system builds positive aspect profiles for each "
        "tourist attraction using ABSA results extracted from "
        "tourist reviews. User preferences and tourist attractions "
        "are represented as aspect vectors, and recommendations are "
        "ranked using cosine similarity. The cosine similarity score "
        "is multiplied by 100 and displayed as Match (%)."
    )