import os
import re
import base64
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

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
    background-color: #f6f8fb;
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
    padding: 14px 18px;
    border-radius: 18px;
    background-color: #eef6ff;
    border: 1px solid #bfdbfe;
    margin-bottom: 22px;
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
    margin-bottom: 12px;
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
    line-height: 1.6;
}

.small-note {
    color: #64748b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Load data / model
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("tourist_profile_data.csv")
    return df


@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


df = load_data()
model = load_sbert_model()


# =========================
# Settings
# =========================
TOP_N = 6

KEYWORDS = {
    "🏛 History": "history",
    "🎎 Culture": "culture",
    "🏯 Architecture": "architecture",
    "📸 Photo Spot": "views",
    "🌳 Nature": "nature",
    "🍜 Food": "food",
    "🛍 Shopping": "shopping",
    "🎡 Activities": "activities",
    "🚇 Easy Access": "accessibility",
    "✨ Clean": "cleanliness",
    "🙅 Less Crowded": "crowds",
    "💸 Budget Friendly": "price",
}

PREFERENCE_TEXT = {
    "history": "rich history and historical places",
    "culture": "traditional culture and cultural experiences",
    "architecture": "beautiful architecture and impressive buildings",
    "views": "great photo spots and scenic views",
    "nature": "natural scenery, parks, and relaxing outdoor spaces",
    "food": "local food, restaurants, and cafes",
    "shopping": "shopping streets, markets, and stores",
    "activities": "fun activities, tours, and experiences",
    "accessibility": "easy access by public transportation",
    "cleanliness": "clean and well-maintained environment",
    "crowds": "quiet and less crowded places",
    "price": "budget-friendly places with good value",
}

DISPLAY_NAME = {
    "history": "Rich history",
    "culture": "Traditional culture",
    "architecture": "Beautiful architecture",
    "views": "Great photo spots",
    "view": "Great photo spots",
    "nature": "Natural scenery",
    "food": "Local food",
    "shopping": "Shopping spots",
    "activities": "Fun experiences",
    "accessibility": "Easy access",
    "cleanliness": "Clean environment",
    "service": "Helpful service",
    "atmosphere": "Nice atmosphere",
    "comfort": "Comfortable visit",
    "family-friendliness": "Family friendly",
    "family_friendliness": "Family friendly",
    "transportation": "Convenient transport",
    "price": "Good value",
    "crowds": "Popular spot",
    "aquarium": "Aquarium",
    "exhibits": "Exhibits",
    "variety_of_fish": "Variety of fish",
}

ASPECT_EMOJI = {
    "history": "🏛",
    "culture": "🎎",
    "architecture": "🏯",
    "views": "📸",
    "view": "📸",
    "nature": "🌳",
    "food": "🍜",
    "shopping": "🛍",
    "activities": "🎡",
    "accessibility": "🚇",
    "transportation": "🚆",
    "cleanliness": "✨",
    "service": "🤝",
    "atmosphere": "🌅",
    "comfort": "🪑",
    "family-friendliness": "👨‍👩‍👧",
    "family_friendliness": "👨‍👩‍👧",
    "price": "💸",
    "crowds": "🔥",
    "aquarium": "🐠",
    "exhibits": "🖼",
    "variety_of_fish": "🐟",
}

TIP_TEXT = {
    "crowds": "This spot can get popular during peak hours, so visiting earlier in the day may give you a more relaxed experience.",
    "price": "Some visitors mention the cost, so it may be worth checking ticket prices or available passes before you go.",
    "waiting_time": "There may be some waiting during busy times, so arriving a little earlier can make the visit smoother.",
    "queue": "Lines can form during peak hours, so planning your visit outside the busiest times is a good idea.",
    "accessibility": "Public transportation or route planning may make the visit easier.",
    "transportation": "Checking the best transit route in advance can help you save time.",
    "cleanliness": "A few visitors mention facility conditions, so it may be helpful to plan short breaks nearby.",
}


# =========================
# Helper functions
# =========================
def clean_place_name(place_name):
    return re.sub(r"^\d+\.\s*", "", str(place_name)).strip()


def get_local_image_path(place_name):
    clean_name = clean_place_name(place_name)

    for ext in ["jpg", "jpeg", "png", "webp"]:
        path = os.path.join("images", f"{clean_name}.{ext}")
        if os.path.exists(path):
            return path

    return None


def image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()

    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"

    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


def show_fixed_image(img_path):
    if img_path:
        img_src = image_to_base64(img_path)
        st.markdown(
            f"""
            <img src="{img_src}" class="fixed-img">
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="image-placeholder">
                Image not available
            </div>
            """,
            unsafe_allow_html=True
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


def pretty_aspect(aspect):
    aspect = str(aspect).strip().lower()
    return DISPLAY_NAME.get(aspect, aspect.replace("_", " ").title())


def build_profile_text(row):
    pos_items = parse_aspects(row.get("positive_top10", ""))
    neg_items = parse_aspects(row.get("negative_top10", ""))

    positive_text = ", ".join([pretty_aspect(a) for a, _ in pos_items[:10]])
    negative_text = ", ".join([pretty_aspect(a) for a, _ in neg_items[:5]])

    return (
        f"This tourist attraction is known for {positive_text}. "
        f"Visitors may also mention {negative_text}."
    )


def build_user_text(selected_keywords):
    selected_texts = [
        PREFERENCE_TEXT[k]
        for k in selected_keywords
        if k in PREFERENCE_TEXT
    ]

    return "I prefer tourist attractions with " + ", ".join(selected_texts) + "."


def highlight_badges(items, limit=5):
    if not items:
        return "<span class='small-note'>No highlight data available</span>"

    html = ""

    for aspect, _ in items[:limit]:
        key = str(aspect).strip().lower()
        emoji = ASPECT_EMOJI.get(key, "✨")

        html += (
            f"<span class='highlight-item'>"
            f"{emoji} {pretty_aspect(aspect)}"
            f"</span>"
        )

    return html


def travel_tip(neg_items):
    if not neg_items:
        return "No major concerns found. Enjoy your visit!"

    for aspect, _ in neg_items:
        key = str(aspect).strip().lower()

        if key in TIP_TEXT:
            return TIP_TEXT[key]

    top_aspect = pretty_aspect(neg_items[0][0]).lower()
    return f"Some visitors mention {top_aspect}, so it may be helpful to check details before your visit."


@st.cache_data
def prepare_profiles(df):
    temp = df.copy()
    temp["profile_text"] = temp.apply(build_profile_text, axis=1)
    return temp


@st.cache_data
def encode_profiles(profile_texts):
    embeddings = model.encode(
        profile_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings


def calculate_sbert_match(selected_keywords, profile_embeddings):
    user_text = build_user_text(selected_keywords)

    user_embedding = model.encode(
        [user_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarities = np.dot(profile_embeddings, user_embedding[0])

    return similarities


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
# Prepare profile embeddings
# =========================
df_profile = prepare_profiles(df)
profile_embeddings = encode_profiles(df_profile["profile_text"].tolist())


# =========================
# Hero
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">🌏 SmartTrip AI</div>
    <div class="hero-sub">
        Discover Korean tourist attractions using ABSA-based destination profiles and SBERT similarity.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Keyword selector
# =========================
st.markdown('<div class="keyword-title">Choose Your Travel Style</div>', unsafe_allow_html=True)
st.caption("Select one or more preferences. Recommendations update automatically.")

cols = st.columns(6)

for i, (label, value) in enumerate(KEYWORDS.items()):
    selected = value in st.session_state.selected_keywords
    btn_label = f"✓ {label}" if selected else label

    with cols[i % 6]:
        if st.button(btn_label, key=f"kw_{value}", use_container_width=True):
            toggle_keyword(value)
            st.rerun()

if st.session_state.selected_keywords:
    selected_text = ", ".join([pretty_aspect(x) for x in st.session_state.selected_keywords])
    st.markdown(
        f"""
        <div class="selected-box">
            <b>Selected Preferences:</b> {selected_text}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Select your travel preferences above to start exploring recommended attractions.")


# =========================
# Recommendation results
# =========================
if st.session_state.selected_keywords:
    similarities = calculate_sbert_match(
        st.session_state.selected_keywords,
        profile_embeddings
    )

    result = df_profile.copy()
    result["similarity"] = similarities
    result["match_percent"] = (result["similarity"] * 100).round().astype(int)
    result["match_percent"] = result["match_percent"].clip(lower=0, upper=100)

    result = result.sort_values("similarity", ascending=False).head(TOP_N)

    st.markdown("## ⭐ Recommended Attractions")

    card_cols = st.columns(2)

    for idx, (_, row) in enumerate(result.iterrows()):
        place_name = clean_place_name(row["location_name"])
        pos_items = parse_aspects(row.get("positive_top10", ""))
        neg_items = parse_aspects(row.get("negative_top10", ""))
        img_path = get_local_image_path(place_name)

        rank = idx + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."

        with card_cols[idx % 2]:
            with st.container(border=True):
                show_fixed_image(img_path)

                st.markdown(
                    f"""
                    <div class="place-title">{medal} {place_name}</div>
                    <div class="match-text">⭐ {row["match_percent"]}% Match</div>
                    """,
                    unsafe_allow_html=True
                )

                if "avg_rating" in row.index and pd.notna(row["avg_rating"]):
                    st.markdown(
                        f"""
                        <div class="rating-text">
                            ⭐ Visitor Rating: {row["avg_rating"]:.2f}/5.0
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown('<div class="section-label">✨ Highlights</div>', unsafe_allow_html=True)
                st.markdown(highlight_badges(pos_items, limit=5), unsafe_allow_html=True)

                st.markdown('<div class="section-label">💡 Good to Know</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="tip-box">
                        {travel_tip(neg_items)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:
    st.markdown("## How it works")
    st.write(
        "This system builds positive and negative aspect profiles for each tourist attraction "
        "using ABSA results extracted from tourist reviews. "
        "User preferences and attraction profiles are represented with SBERT embeddings, "
        "and recommendations are ranked by cosine similarity."
    )