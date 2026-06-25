from dotenv import load_dotenv
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_PATH = "data/tripadvisor.csv"
TEXT_COL = "review_text"

MODEL_NAME = "gpt-4.1-mini"

SAVE_EVERY = 1000
CHUNK_SIZE = 100
MAX_WORKERS = 5

load_dotenv(r"C:\Users\Avy012\Desktop\ABSA\.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30,
    max_retries=0
)


def analyze_review(review_text, max_retries=2):
    prompt = f"""
You will be provided with a tourist attraction review.
A review sentence usually covers the customer opinions expressed on different aspects of a tourist attraction.

You are tasked to perform Aspect-based Sentiment Analysis to extract the user sentiments expressed on different aspects in the review.

- Aspect Extraction: Identifying aspect targets from opinionated text. All extracted aspect names must be written in English.
- Aspect Sentiment Classification: From the extracted aspect target, predict the sentiment polarity of user opinions on the aspect.

If an aspect is expressed in a language other than English, translate it into English before returning the result.

Return JSON only.
Format:
{{
  "aspects": [
    {{"aspect": "view", "sentiment": "positive"}},
    {{"aspect": "crowding", "sentiment": "negative"}}
  ]
}}

Review: {review_text}
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"\n[GPT ERROR] {repr(e)} / retry after {wait}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("GPT failed after retries")


def normalize_output(output):
    if "aspects" in output:
        return output["aspects"]

    if "aspect" in output and "sentiment" in output:
        if isinstance(output["aspect"], list):
            return [
                {"aspect": a, "sentiment": s}
                for a, s in zip(output["aspect"], output["sentiment"])
            ]
        else:
            return [{"aspect": output["aspect"], "sentiment": output["sentiment"]}]

    return []


def process_row(row_data):
    idx, row = row_data
    review_text = row[TEXT_COL]

    try:
        output = analyze_review(review_text)
        absa_results = normalize_output(output)

        if not absa_results:
            return [{
                "index": idx,
                "reviewID": row["ReviewID"],
                "review_text": review_text,
                "aspect": None,
                "sentiment": None,
                "status": "no_aspect"
            }]

        results = []

        for item in absa_results:
            aspect = (
                str(item.get("aspect", ""))
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            sentiment = str(item.get("sentiment", "")).strip().lower()

            results.append({
                "index": idx,
                "reviewID": row["ReviewID"],
                "review_text": review_text,
                "aspect": aspect,
                "sentiment": sentiment,
                "status": "success"
            })

        return results

    except Exception as e:
        return [{
            "index": idx,
            "reviewID": row["ReviewID"],
            "review_text": review_text,
            "aspect": None,
            "sentiment": None,
            "status": f"error: {repr(e)}"
        }]


df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=[TEXT_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)

print("Total reviews:", len(df), flush=True)

rows = list(df.iterrows())

OUTPUT_DIR = "file"
os.makedirs(OUTPUT_DIR, exist_ok=True)

error_count = 0
total_processed_count = 0

START_INDEX = 0

for save_start in range(START_INDEX, len(rows), SAVE_EVERY):

    save_end = min(save_start + SAVE_EVERY, len(rows))
    save_rows = rows[save_start:save_end]

    chunk_results = []
    chunk_processed_count = 0

    print(f"\nSave block: {save_start} ~ {save_end - 1}", flush=True)

    for chunk_start in range(save_start, save_end, CHUNK_SIZE):

        print(f"Chunk start: {chunk_start}", flush=True)

        chunk = rows[chunk_start:min(chunk_start + CHUNK_SIZE, save_end)]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_row, row_data)
                for row_data in chunk
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):

                result_rows = future.result()

                if result_rows[0]["status"] not in ["success", "no_aspect"]:
                    error_count += 1
                    print(result_rows[0]["status"], flush=True)

                chunk_results.extend(result_rows)
                chunk_processed_count += 1
                total_processed_count += 1

    output_path = os.path.join(
        OUTPUT_DIR,
        f"review_absa_4omini_{save_start:06d}_{save_end - 1:06d}.csv"
    )

    pd.DataFrame(chunk_results).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved: {output_path}", flush=True)
    print(
        f"Saved block reviews: {chunk_processed_count} / Total processed: {total_processed_count} / errors: {error_count}",
        flush=True
    )

print("All chunk files saved.")
print("Total processed:", total_processed_count)
print("Total errors:", error_count)

## 파일 합치기

all_files = sorted([
    os.path.join(OUTPUT_DIR, f)
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".csv")
])

full_aspect = pd.concat(
    [pd.read_csv(f) for f in all_files],
    ignore_index=True
)

full_aspect.to_csv(
    "data/full_aspect.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Merged full aspect saved: data/full_aspect.csv")