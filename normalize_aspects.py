import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Paths
# =========================
ABSA_CSV = "data/full_aspect.csv"
TRIPADVISOR_CSV = "data/tripadvisor.csv"
METADATA_CSV = "data/tourist_metadata.csv"

PROFILE_CSV = "data/profile.csv"
NORMALIZED_CSV = "data/tourist_profile_normalized.csv"
MAP_OUTPUT = "data/aspect_category_map.csv"
FINAL_PROFILE_CSV = "data/tourist_profile_data.csv"


# =========================
# GPT Settings
# =========================
MODEL_NAME = "gpt-4.1-mini"
CHUNK_SIZE = 50
SAVE_EVERY_CHUNKS = 5

load_dotenv(r"C:\Users\Avy012\Desktop\ABSA\.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Canonical Categories
# =========================
CATEGORIES = [
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
    "Others"
]


SYSTEM_PROMPT = f"""
You are an expert in tourism review analysis.

Your task is to assign each tourist aspect to exactly one canonical tourism aspect category.

Allowed categories:
{CATEGORIES}

Guidelines:
- Choose the single category that best represents the main meaning.
- Ignore location names and unnecessary modifiers.
- Group synonymous expressions into the same category.
- If none of the categories clearly applies, use "Others".
- Output only valid JSON.
- JSON format must be:
{{
  "aspect_name": "Category",
  "aspect_name": "Category"
}}
""".strip()


# =========================
# Step 1. Build profile.csv
# =========================
def build_profile_from_absa():
    print("Building profile.csv from ABSA result...")

    aspect = pd.read_csv(ABSA_CSV)
    og = pd.read_csv(TRIPADVISOR_CSV)

    aspect = aspect.dropna(subset=["reviewID", "aspect", "sentiment"]).copy()

    aspect["reviewID"] = aspect["reviewID"].astype(str)
    og["ReviewID"] = og["ReviewID"].astype(str)

    aspect["aspect"] = (
        aspect["aspect"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    aspect["sentiment"] = (
        aspect["sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    aspect = aspect[
        aspect["sentiment"].isin(["positive", "negative", "neutral"])
    ].copy()

    merged = og[["ReviewID", "location_name"]].merge(
        aspect,
        left_on="ReviewID",
        right_on="reviewID",
        how="inner"
    )

    profile = (
        merged.groupby(["location_name", "aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    os.makedirs(os.path.dirname(PROFILE_CSV), exist_ok=True)

    profile.to_csv(
        PROFILE_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Profile saved:", PROFILE_CSV)


# =========================
# Step 2. Normalize aspects
# =========================
def extract_json(text):
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found")

    return json.loads(match.group())


def gpt_normalize_chunk(aspects, max_retries=3):
    aspect_text = "\n".join([f"- {a}" for a in aspects])

    user_prompt = f"""
Normalize the following tourist aspects.

Aspects:
{aspect_text}
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                temperature=0,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )

            result = extract_json(response.output_text)

            cleaned = {}

            for aspect in aspects:
                category = result.get(aspect, "Others")

                if category not in CATEGORIES:
                    category = "Others"

                cleaned[aspect] = category

            return cleaned

        except Exception as e:
            print(f"[Error] chunk attempt {attempt + 1}/{max_retries} | {e}")
            time.sleep(3)

    return {aspect: "Others" for aspect in aspects}


def save_mapping(aspect_category_map):
    pd.DataFrame({
        "aspect": list(aspect_category_map.keys()),
        "canonical_aspect": list(aspect_category_map.values())
    }).to_csv(
        MAP_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


def normalize_aspects():
    if not os.path.exists(PROFILE_CSV):
        build_profile_from_absa()

    profile = pd.read_csv(PROFILE_CSV)

    profile["aspect"] = (
        profile["aspect"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    profile["count"] = pd.to_numeric(
        profile["count"],
        errors="coerce"
    ).fillna(0)

    aspect_counts = (
        profile
        .dropna(subset=["aspect"])
        .groupby("aspect")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    unique_aspects = (
        aspect_counts[
            aspect_counts >= 5
        ]
        .index
        .tolist()
    )

    print("Total unique aspects :", len(aspect_counts))
    print("Aspects to normalize :", len(unique_aspects))
    print("Rare aspects (<5) :", (aspect_counts < 5).sum())

    if os.path.exists(MAP_OUTPUT):
        existing_map_df = pd.read_csv(MAP_OUTPUT)

        aspect_category_map = dict(
            zip(
                existing_map_df["aspect"],
                existing_map_df["canonical_aspect"]
            )
        )

        print("Existing mappings loaded :", len(aspect_category_map))

    else:
        aspect_category_map = {}

    remaining_aspects = [
        aspect for aspect in unique_aspects
        if aspect not in aspect_category_map
    ]

    print("Remaining aspects :", len(remaining_aspects))

    chunks = [
        remaining_aspects[i:i + CHUNK_SIZE]
        for i in range(0, len(remaining_aspects), CHUNK_SIZE)
    ]

    for chunk_idx, aspect_chunk in enumerate(tqdm(chunks), start=1):
        chunk_result = gpt_normalize_chunk(aspect_chunk)
        aspect_category_map.update(chunk_result)

        if chunk_idx % SAVE_EVERY_CHUNKS == 0:
            save_mapping(aspect_category_map)
            print(f"Saved mapping at chunk {chunk_idx}")

    save_mapping(aspect_category_map)

    profile["canonical_aspect"] = (
        profile["aspect"]
        .map(aspect_category_map)
        .fillna("Others")
    )

    profile.to_csv(
        NORMALIZED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Normalized profile saved:", NORMALIZED_CSV)


# =========================
# Step 3. Build final tourist profile
# =========================
def build_tourist_profile_data():
    print("Building final tourist profile data...")

    norm = pd.read_csv(NORMALIZED_CSV)
    og = pd.read_csv(TRIPADVISOR_CSV)
    metadata = pd.read_csv(METADATA_CSV)

    norm = norm[norm["canonical_aspect"].str.lower() != "others"].copy()

    # 동일한 canonical aspect로 매핑된 원시 aspect들의 count를 먼저 합산
    aggregated = (
        norm
        .groupby(
            ["location_name", "canonical_aspect", "sentiment"],
            as_index=False
        )["count"]
        .sum()
    )


    # 관광지별 positive 상위 10개 canonical aspect
    positive = (
        aggregated[
            aggregated["sentiment"] == "positive"
        ]
        .sort_values(
            ["location_name", "count"],
            ascending=[True, False]
        )
        .groupby("location_name")
    )


    # 관광지별 negative 상위 10개 canonical aspect
    negative = (
        aggregated[
            aggregated["sentiment"] == "negative"
        ]
        .sort_values(
            ["location_name", "count"],
            ascending=[True, False]
        )
        .groupby("location_name")
    )

    # 관광지별 모든 positive canonical aspect
    positive = (
        aggregated[
            aggregated["sentiment"] == "positive"
        ]
        .sort_values(
            ["location_name", "count"],
            ascending=[True, False]
        )
        .copy()
    )


    # 관광지별 모든 negative canonical aspect
    negative = (
        aggregated[
            aggregated["sentiment"] == "negative"
        ]
        .sort_values(
            ["location_name", "count"],
            ascending=[True, False]
        )
        .copy()
    )


    # 관광지별 positive aspect를 하나의 문자열로 저장
    positive_summary = (
        positive
        .groupby("location_name")
        .apply(
            lambda x: ", ".join(
                f"{aspect}({count})"
                for aspect, count in zip(
                    x["canonical_aspect"],
                    x["count"]
                )
            )
        )
        .reset_index(name="positive")
    )


    # 관광지별 negative aspect를 하나의 문자열로 저장
    negative_summary = (
        negative
        .groupby("location_name")
        .apply(
            lambda x: ", ".join(
                f"{aspect}({count})"
                for aspect, count in zip(
                    x["canonical_aspect"],
                    x["count"]
                )
            )
        )
        .reset_index(name="negative")
    )


    tourist_profile = positive_summary.merge(
        negative_summary,
        on="location_name",
        how="outer"
    )

    remove_locations = [
        "61. MJ Jeju Diving Club",
        "63. Seoul Free Walking Tour",
        "57. Seoul Pub Crawl",
        "41. Seoul Sky",
        "40. Seoul Metro"
    ]

    tourist_profile = tourist_profile[
        ~tourist_profile["location_name"].isin(remove_locations)
    ].reset_index(drop=True)

    rating_mean = (
        og.groupby("location_name")["review_rating"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"review_rating": "avg_rating"})
    )

    tourist_profile = tourist_profile.merge(
        rating_mean,
        on="location_name",
        how="left"
    )

    required_metadata_columns = [
        "location_name",
        "english_address",
        "intro",
        "OWI",
        "topic",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "Missing columns in tourist_metadata.csv: "
            + ", ".join(missing_metadata_columns)
        )

    metadata = metadata[required_metadata_columns].copy()

    if metadata["location_name"].duplicated().any():
        duplicated_names = metadata.loc[
            metadata["location_name"].duplicated(keep=False),
            "location_name"
        ].tolist()

        raise ValueError(
            f"Duplicate location_name values in metadata: {duplicated_names}"
        )

    tourist_profile = tourist_profile.merge(
        metadata,
        on="location_name",
        how="left",
        validate="one_to_one"
    )

    tourist_profile.to_csv(
        FINAL_PROFILE_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Final tourist profile saved:", FINAL_PROFILE_CSV)


# =========================
# Main
# =========================
if __name__ == "__main__":
    normalize_aspects()
    build_tourist_profile_data()
    print("Done.")