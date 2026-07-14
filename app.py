import os
import re
import base64
import html
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
    background:#EFF6FF;
    border:1px solid #BFDBFE;
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 12px;
    margin-bottom: 22px;
    color:#1E3A5F;
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
    color:#374151;
    line-height:1.6;
    margin-bottom:12px;
    font-size:15px;

    height: 72px;
    overflow-y: auto;
    padding-right: 4px;
}

.address-text {
    color:#6B7280;
    font-size:14px;
    line-height:1.5;
    margin-bottom:14px;
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
# Load data / model
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/tourist_profile_data.csv")

@st.cache_resource(show_spinner="Loading recommendation model...")
def load_sbert_model():
    hf_token = st.secrets.get("HF_TOKEN", None)

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token,
        device="cpu"
    )


df = load_data()

print("before model")
model = load_sbert_model()
print("after model")


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
# Helper functions
# =========================
def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def build_profile_text(row):
    pos_items = parse_aspects(row.get("positive_top10", ""))
    neg_items = parse_aspects(row.get("negative_top10", ""))

    positive_text = ", ".join([aspect for aspect, _ in pos_items[:10]])
    negative_text = ", ".join([aspect for aspect, _ in neg_items[:5]])
    intro = safe_text(row.get("intro", ""))

    return (
        f"{intro} "
        f"This tourist attraction is known for {positive_text}. "
        f"Visitors may also mention {negative_text}."
    )


def build_user_text(selected_keywords):
    return "I prefer tourist attractions with " + ", ".join(selected_keywords) + "."


def highlight_badges(items, limit=5):
    if not items:
        return "<span class='small-note'>No highlight data available</span>"

    html_code = ""

    for aspect, _ in items[:limit]:
        aspect = safe_text(aspect)
        emoji = ASPECT_EMOJI.get(aspect, "✨")

        html_code += (
            f"<span class='highlight-item'>"
            f"{emoji} {html.escape(aspect)}"
            f"</span>"
        )

    return html_code


def get_good_to_know(row):
    return safe_text(row.get("good_to_know", ""))


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


def calculate_sbert_match(selected_keywords, profile_embeddings, df_profile):
    user_text = build_user_text(selected_keywords)

    user_embedding = model.encode(
        [user_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    sbert_sim = np.dot(profile_embeddings, user_embedding[0])

    selected_set = set(selected_keywords)

    aspect_scores = []

    for _, row in df_profile.iterrows():
        pos_items = parse_aspects(row.get("positive_top10", ""))
        pos_aspects = {aspect for aspect, _ in pos_items}

        overlap = len(selected_set & pos_aspects)
        overlap_ratio = overlap / len(selected_set) if selected_set else 0

        aspect_scores.append(overlap_ratio)

    aspect_scores = np.array(aspect_scores)

    final_scores = (0.35 * sbert_sim) + (0.65 * aspect_scores)

    return final_scores


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
        Discover Must-Visit Attractions in South Korea.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Keyword selector
# =========================
st.markdown(
    '<div class="keyword-title">Choose Your Travel Style</div>',
    unsafe_allow_html=True
)
st.caption("Select one or more preferences. Recommendations update automatically.")

cols = st.columns(5)

for i, aspect in enumerate(KEYWORDS):
    emoji = ASPECT_EMOJI.get(aspect, "✨")
    selected = aspect in st.session_state.selected_keywords
    btn_label = f"✓ {emoji} {aspect}" if selected else f"{emoji} {aspect}"

    with cols[i % 5]:
        if st.button(btn_label, key=f"kw_{aspect}", use_container_width=True):
            toggle_keyword(aspect)
            st.rerun()

if st.session_state.selected_keywords:
    selected_text = ", ".join(st.session_state.selected_keywords)
    st.markdown(
        f"""
        <div class="selected-box">
            <b>Selected Preferences:</b> {html.escape(selected_text)}
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
    profile_embeddings,
    df_profile
)

    result = df_profile.copy()
    result["similarity"] = similarities
    result = result.sort_values("similarity", ascending=False).head(TOP_N)

    sim_min = result["similarity"].min()
    sim_max = result["similarity"].max()

    if sim_max == sim_min:
        result["match_percent"] = 85
    else:
        result["match_percent"] = (
            70 + (result["similarity"] - sim_min) / (sim_max - sim_min) * 28
        ).round().astype(int)

    result["match_percent"] = result["match_percent"].clip(70, 98)

    st.markdown("## ⭐ Recommended Attractions")

    card_cols = st.columns(2)

    for idx, (_, row) in enumerate(result.iterrows()):
        place_name = clean_place_name(row["location_name"])
        pos_items = parse_aspects(row.get("positive_top10", ""))
        img_path = get_local_image_path(place_name)

        intro = safe_text(row.get("intro", ""))
        address = safe_text(row.get("english_address", ""))
        good_to_know = get_good_to_know(row)

        rank = idx + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."

        with card_cols[idx % 2]:
            with st.container(border=True):
                show_fixed_image(img_path)

                st.markdown(
                    f"""
                    <div class="place-title">{medal} {html.escape(place_name)}</div>
                    <div class="match-text">⭐ {row["match_percent"]}% Match</div>
                    """,
                    unsafe_allow_html=True
                )

                if "avg_rating" in row.index and pd.notna(row["avg_rating"]):
                    st.markdown(
                        f"""
                        <div class="rating-text">
                            ⭐ Visitor Rating: {float(row["avg_rating"]):.1f}/5.0
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if intro:
                    st.markdown(
                        f"""
                        <div class="intro-text">
                            {html.escape(intro)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if address:
                    st.markdown(
                        f"""
                        <div class="address-text">
                            📍 {html.escape(address)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="section-label">✨ Highlights</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    highlight_badges(pos_items, limit=5),
                    unsafe_allow_html=True
                )

                if good_to_know:
                    st.markdown(
                        '<div class="section-label">💡 Good to Know</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div class="tip-box">
                            {html.escape(good_to_know)}
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