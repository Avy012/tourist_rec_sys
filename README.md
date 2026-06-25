# Streamlit 웹사이트 주소
```https://touristrecsys-ot5u6kujx4nkyzt2p34zwe.streamlit.app/ ```

# 프로젝트 구조

| 파일 | 설명 |
|------|------|
| **app.py** | **Streamlit 메인 애플리케이션**입니다. 사용자의 선호 측면(aspect)을 입력받아 SBERT 유사도를 기반으로 관광지를 추천하고 결과를 시각화합니다. |
| **gpt_4_1_mini.py** | GPT-4.1 mini를 이용하여 TripAdvisor 리뷰에서 **Aspect-Based Sentiment Analysis (ABSA)** 를 수행하고 aspect와 sentiment를 추출합니다. |
| **normalize_aspects.py** | 다양한 aspect 표현을 대표 카테고리(canonical aspect)로 정규화합니다. |
| **generate_good_to_know.py** | 부정 리뷰에서 자주 언급된 내용을 바탕으로 관광지별 **Good to Know** 정보를 생성합니다. |
| **tourist_adress_with_intro.csv** | 수작업으로 수집한 53개 관광지에 대한 영문 주소 및 관광지 소개 데이터입니다. |
| **tourist_profile_data.csv** | 최종 관광지 프로필 데이터입니다. Intro, 대표 장점, Good to Know, 평균 평점 등 추천에 필요한 정보를 포함합니다. |
| **images/** | 관광지 이미지를 저장하는 폴더입니다. |
| **.streamlit/** | Streamlit 실행 및 UI 설정 파일을 저장하는 폴더입니다. |
| **requirements.txt** | 프로젝트 실행에 필요한 Python 라이브러리 목록입니다. |

---

# 실행 방법

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. ABSA 추출 및 관광지 프로필 만들기

```bash
python gpt_4_1_mini.py
python normalize_aspects.py
python generate_good_to_know.py
```

### 3. Streamlit 실행

```bash
streamlit run app.py
```