import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

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

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    hf_token = st.secrets.get("HF_TOKEN", None)

    cache_dir = "/tmp/huggingface_sbert_cache"
    os.makedirs(cache_dir, exist_ok=True)

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token,
        device="cpu",
        cache_folder=cache_dir
    )


df = load_data()


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


def validate_profile_data(df_profile):
    """Validate the uploaded attraction-profile data before SBERT encoding."""
    required_columns = [
        "location_name",
        "positive_top10",
        "negative_top10",
        "intro",
    ]

    missing_columns = [
        column for column in required_columns
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

    if "profile_text" not in df_profile.columns:
        raise ValueError("profile_text was not generated.")

    null_profile_count = int(df_profile["profile_text"].isna().sum())
    empty_profile_count = int(
        df_profile["profile_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    profile_lengths = (
        df_profile["profile_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    return {
        "rows": len(df_profile),
        "columns": df_profile.columns.tolist(),
        "null_profile_count": null_profile_count,
        "empty_profile_count": empty_profile_count,
        "min_profile_length": int(profile_lengths.min()),
        "max_profile_length": int(profile_lengths.max()),
        "mean_profile_length": float(profile_lengths.mean()),
    }


@st.cache_data(show_spinner=False)
def encode_profiles(profile_texts):
    """Encode attraction profiles after the model has been loaded lazily."""
    model = load_sbert_model()

    embeddings = model.encode(
        list(profile_texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings


def calculate_sbert_match(selected_keywords, profile_embeddings, df_profile):
    model = load_sbert_model()
    user_text = build_user_text(selected_keywords)

    user_embedding = model.encode(
        [user_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
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
    status_box = st.status(
        "Preparing recommendations...",
        expanded=True
    )

    try:
        with status_box:
            st.write("Step 1/5: Validating tourist profile data...")
            print("Step 1/5: Validating tourist profile data", flush=True)

            validation = validate_profile_data(df_profile)

            st.write(f"Rows: {validation['rows']}")
            st.write(
                "Profile text length: "
                f"min={validation['min_profile_length']}, "
                f"mean={validation['mean_profile_length']:.1f}, "
                f"max={validation['max_profile_length']}"
            )

            if validation["null_profile_count"] > 0:
                st.warning(
                    f"{validation['null_profile_count']} profile_text values are null."
                )

            if validation["empty_profile_count"] > 0:
                st.warning(
                    f"{validation['empty_profile_count']} profile_text values are empty."
                )

            if validation["max_profile_length"] > 10000:
                st.warning(
                    "At least one profile_text is unusually long. "
                    "This may slow down encoding."
                )

            st.write("Step 2/5: Loading SBERT model...")
            print("Step 2/5: Loading SBERT model", flush=True)

            model = load_sbert_model()

            st.write("Step 2/5 complete: SBERT model loaded.")
            print("Step 2/5 complete: SBERT model loaded", flush=True)

            profile_texts = tuple(
                df_profile["profile_text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            st.write(
                f"Step 3/5: Encoding {len(profile_texts)} attraction profiles..."
            )
            print(
                f"Step 3/5: Encoding {len(profile_texts)} attraction profiles",
                flush=True
            )

            profile_embeddings = encode_profiles(profile_texts)

            st.write(
                "Step 3/5 complete: "
                f"embedding shape={profile_embeddings.shape}"
            )
            print(
                f"Step 3/5 complete: shape={profile_embeddings.shape}",
                flush=True
            )

            st.write("Step 4/5: Encoding user preferences...")
            print("Step 4/5: Encoding user preferences", flush=True)

            similarities = calculate_sbert_match(
                st.session_state.selected_keywords,
                profile_embeddings,
                df_profile
            )

            st.write("Step 4/5 complete: Similarities calculated.")
            print("Step 4/5 complete: Similarities calculated", flush=True)

            st.write("Step 5/5: Building recommendation cards...")
            print("Step 5/5: Building recommendation cards", flush=True)

        status_box.update(
            label="Recommendations ready",
            state="complete",
            expanded=False
        )

    except Exception as exc:
        status_box.update(
            label="Recommendation failed",
            state="error",
            expanded=True
        )
        st.error(f"{type(exc).__name__}: {exc}")
        print(
            f"Recommendation error: {type(exc).__name__}: {exc}",
            flush=True
        )
        st.stop()

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