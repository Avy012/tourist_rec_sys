# 관광지 추천 시스템 (Tourist Recommendation System)

## Streamlit 웹사이트

https://touristrecsys-sfi47ztbuvhvpopq5gin9k.streamlit.app/

---

# 프로젝트 구조

| 파일 | 설명 |
|------|------|
| **app.py** | Streamlit 기반 관광지 추천 웹 애플리케이션입니다. 사용자가 선호하는 관광 요소(Aspect)를 선택하면 관광지 프로필과의 Cosine Similarity를 계산하여 관광지를 추천합니다. |
| **gpt_4_1_mini.py** | GPT-4.1 mini를 이용하여 TripAdvisor 리뷰에 대해 Aspect-Based Sentiment Analysis(ABSA)를 수행하고 aspect와 sentiment를 추출합니다. |
| **normalize_aspects.py** | 다양한 aspect 표현을 대표 관광 카테고리(Canonical Aspect)로 정규화하고 관광지 프로필을 생성합니다. |
| **generate_good_to_know.py** | 부정 리뷰에서 자주 언급되는 내용을 기반으로 관광지별 **Good to Know** 정보를 생성합니다. |
| **tourist_metadata.csv** | 관광지 메타데이터입니다. 관광지 소개, 영문 주소, OWI, Visitor Insights 정보를 포함하며 전처리의 입력 데이터로 사용됩니다. |
| **tourist_profile_data_with_tips.csv** | 추천 시스템에서 사용하는 최종 관광지 프로필입니다. 관광지 프로필, 메타데이터, Good to Know 정보를 포함합니다. |
| **images/** | 관광지 이미지를 저장하는 폴더입니다. |
| **.streamlit/** | Streamlit 설정 파일입니다. |
| **requirements.txt** | 프로젝트 실행에 필요한 Python 라이브러리 목록입니다. |

---

# 데이터셋

## 1. 초기 데이터 (Input Dataset)

### tourist_metadata.csv

53개 관광지의 메타데이터를 저장한 파일입니다.

이 데이터는 다음 정보를 포함합니다.

- 관광지 이름
- 영문 주소
- 관광지 소개
- OWI (Overcrowding Warning Index)
- Visitor Insights (Positive / Neutral / Negative 대표 Topic)

| 컬럼 | 설명 |
|------|------|
| location_name | 관광지 이름 |
| english_address | 영문 주소 |
| intro | 관광지 소개 |
| OWI | 혼잡도 지수 (Overcrowding Warning Index) |
| topic | 감성별 대표 Topic 및 Display Phrase (JSON 형태) |

---

## 2. 생성 데이터 (Generated Dataset)

### ① full_aspect.csv

생성 코드

```text
gpt_4_1_mini.py
```

TripAdvisor 리뷰에 대해 GPT-4.1 mini를 이용하여 수행한 ABSA 결과입니다.

---

### ② profile.csv

생성 코드

```text
normalize_aspects.py
```

관광지별 aspect 및 sentiment 빈도를 집계한 데이터입니다.

---

### ③ aspect_category_map.csv

생성 코드

```text
normalize_aspects.py
```

다양한 aspect 표현을 대표 관광 카테고리(Canonical Aspect)로 정규화한 매핑 테이블입니다.

---

### ④ tourist_profile_normalized.csv

생성 코드

```text
normalize_aspects.py
```

Canonical Aspect가 적용된 관광지 프로필입니다.

---

### ⑤ tourist_profile_data.csv

생성 코드

```text
normalize_aspects.py
```

추천 시스템에서 사용하는 관광지 프로필 데이터입니다.

---

### ⑥ tourist_profile_data_with_tips.csv

생성 코드

```text
generate_good_to_know.py
```

최종 추천 시스템에서 사용하는 데이터입니다.
Good to Know 정보가 추가됩니다.

---

# 실행 방법

## 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

## 2. 관광지 프로필 생성

아래 순서대로 실행합니다.

```bash
python gpt_4_1_mini.py
python normalize_aspects.py
python generate_good_to_know.py
```

생성된 데이터는 `data/` 폴더에 저장됩니다.

---

## 3. Streamlit 실행

```bash
streamlit run app.py
```

---
