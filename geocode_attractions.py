"""One-time, cached coordinate lookup; never called by the Streamlit app.

Follow https://operations.osmfoundation.org/policies/nominatim/:
one machine/thread, at most one request per second, cache every response.
"""
import csv
import json
import re
import time
from pathlib import Path

import requests

CACHE = Path("data/geocoding_cache.json")
OUTPUT = Path("data/attraction_coordinates.csv")
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
session = requests.Session()
session.headers["User-Agent"] = "SmartTripAI/1.0 (https://github.com/Avy012/tourist_rec_sys)"
last_request = 0.0
ALIASES = {
    "N Seoul Tower": "남산서울타워",
    "Seongsan Ilchulbong": "성산일출봉",
    "Hwaseong Fortress": "수원화성",
    "DMZ": "제3땅굴",
    "The War and Women's Human Rights Museum": "전쟁과 여성인권 박물관",
    "Bugak Palgakjeong Pavilion": "북악 팔각정",
    "The Spa Grand Hyatt Seoul": "Grand Hyatt Seoul",
    "Starfield COEX Mall": "스타필드 코엑스몰",
    "Lotte World Tower & Mall": "롯데월드타워",
    "Sea Life Coex Seoul Aquarium": "코엑스 아쿠아리움",
    "Namsangol Hanok Village": "남산골한옥마을",
    "Jeonju Hanok Village": "Jeonju Hanok Village",
    "Insadong": "인사동 서울",
    "Songdo Cable Car": "송도해상케이블카",
    "Daepo Haean Jusangjeolli Cliff": "대포주상절리",
}


def lookup(query):
    global last_request
    if query not in cache:
        time.sleep(max(0, 1.1 - (time.monotonic() - last_request)))
        response = session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"format": "jsonv2", "q": query, "countrycodes": "kr", "limit": 1},
            timeout=25,
        )
        last_request = time.monotonic()
        response.raise_for_status()
        cache[query] = response.json()
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache[query]


rows = []
with Path("data/tourist_profile_data_with_tips.csv").open(encoding="utf-8-sig", newline="") as stream:
    profiles = list(csv.DictReader(stream))
for index, profile in enumerate(profiles, 1):
    name = re.sub(r"^\s*\d+\s*\.\s*", "", profile["location_name"]).strip()
    candidates = lookup(ALIASES.get(name, name))
    precision = "attraction"
    if not candidates:
        candidates = lookup(profile["english_address"].strip(", "))
        precision = "address"
        if candidates and candidates[0].get("type") not in {"house", "building"}:
            candidates = []
    if candidates:
        item = candidates[0]
        rows.append({
            "location_name": profile["location_name"],
            "latitude": item["lat"], "longitude": item["lon"],
            "source": f"https://www.openstreetmap.org/{item['osm_type']}/{item['osm_id']}",
            "matched_name": item.get("display_name", ""),
            "precision": precision,
        })
    print(f"{index}/{len(profiles)} {name}: {'found' if candidates else 'unresolved'}", flush=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["location_name", "latitude", "longitude", "source", "matched_name", "precision"])
        writer.writeheader()
        writer.writerows(rows)
