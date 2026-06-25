import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

INPUT_CSV = "./recsys/tourist_profile_data.csv"
OUTPUT_CSV = "tourist_profile_data_with_tips.csv"

MODEL_NAME = "gpt-4.1-mini"
CHUNK_SIZE = 10
SAVE_EVERY_CHUNKS = 3

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def extract_json(text):
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    return json.loads(match.group())


SYSTEM_PROMPT = """
You are writing friendly travel tips for a tourism recommendation app.

For each place, write a "Good to Know" tip based on its less positive review aspects.

Rules:
- Write 2 to 3 natural sentences for each place.
- Do not sound too negative or alarming.
- Do not mention aspect counts such as "(12)".
- Do not say "negative aspects".
- Make it sound helpful, cute, and travel-friendly.
- Use English only.
- Output only valid JSON.
- JSON format must be:
{
  "row_index": "Good to Know text",
  "row_index": "Good to Know text"
}
""".strip()


def generate_good_to_know_chunk(rows, max_retries=3):
    items = []

    for idx, location_name, negative_top10 in rows:
        negative_top10 = clean_text(negative_top10)

        if negative_top10 == "":
            items.append({
                "row_index": str(idx),
                "location_name": location_name,
                "negative_top10": "None"
            })
        else:
            items.append({
                "row_index": str(idx),
                "location_name": location_name,
                "negative_top10": negative_top10
            })

    user_prompt = f"""
Create Good to Know tips for the following places.

Places:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_output_tokens=1200,
            )

            result = extract_json(response.output_text)

            cleaned = {}

            for idx, location_name, negative_top10 in rows:
                idx_str = str(idx)

                default_tip = (
                    "Most visitors describe this place positively overall. "
                    "It is still a good idea to check the location, opening hours, and nearby transportation before visiting."
                )

                tip = clean_text(result.get(idx_str, default_tip))

                if tip == "":
                    tip = default_tip

                cleaned[idx] = tip

            return cleaned

        except Exception as e:
            print(f"[Error] chunk attempt {attempt + 1}/{max_retries} | {e}")
            time.sleep(3)

    return {
        idx: (
            "Most visitors describe this place positively overall. "
            "It is still a good idea to check the location, opening hours, and nearby transportation before visiting."
        )
        for idx, location_name, negative_top10 in rows
    }


df = pd.read_csv(INPUT_CSV)

if "good_to_know" not in df.columns:
    df["good_to_know"] = ""

remaining_rows = []

for i, row in df.iterrows():
    existing = clean_text(row.get("good_to_know", ""))

    if existing:
        continue

    remaining_rows.append((
        i,
        row["location_name"],
        row.get("negative_top10", "")
    ))

print("Total rows:", len(df))
print("Remaining rows:", len(remaining_rows))

chunks = [
    remaining_rows[i:i + CHUNK_SIZE]
    for i in range(0, len(remaining_rows), CHUNK_SIZE)
]

for chunk_idx, chunk in enumerate(tqdm(chunks), start=1):

    result = generate_good_to_know_chunk(chunk)

    for row_idx, tip in result.items():
        df.at[row_idx, "good_to_know"] = tip

    if chunk_idx % SAVE_EVERY_CHUNKS == 0:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved at chunk {chunk_idx}")

    time.sleep(0.5)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_CSV)