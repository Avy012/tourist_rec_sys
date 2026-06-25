import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

INPUT_CSV = "profile.csv"
OUTPUT_CSV = "tourist_profile_normalized.csv"
MAP_OUTPUT = "aspect_category_map.csv"

MODEL_NAME = "gpt-4.1-mini"
CHUNK_SIZE = 50
SAVE_EVERY_CHUNKS = 5

load_dotenv(r"C:\Users\Avy012\Desktop\ABSA\.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

CATEGORIES = [
    "Scenery", "Nature", "History & Heritage", "Culture", "Architecture",
    "Food & Drink", "Shopping", "Activities", "Transportation",
    "Accessibility", "Facilities", "Service", "Price", "Crowds",
    "Cleanliness", "Safety", "Atmosphere", "Others"
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
    }).to_csv(MAP_OUTPUT, index=False, encoding="utf-8-sig")


profile = pd.read_csv(INPUT_CSV)

profile["aspect"] = (
    profile["aspect"]
    .astype(str)
    .str.lower()
    .str.strip()
)

aspect_counts = profile["aspect"].dropna().value_counts()

unique_aspects = aspect_counts[aspect_counts >= 5].index.tolist()

print("Total unique aspects :", len(aspect_counts))
print("Aspects to normalize :", len(unique_aspects))
print("Rare aspects (<5) :", (aspect_counts < 5).sum())

if os.path.exists(MAP_OUTPUT):
    existing_map_df = pd.read_csv(MAP_OUTPUT)
    aspect_category_map = dict(
        zip(existing_map_df["aspect"], existing_map_df["canonical_aspect"])
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

profile.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Done.")
print("Profile saved :", OUTPUT_CSV)