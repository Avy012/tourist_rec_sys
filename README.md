# 프로젝트 구조

### app.py
Streamlit 기반 관광지 추천 웹 애플리케이션의 메인 실행 파일입니다.  
사용자의 선호 측면(aspect)을 입력받아 SBERT 유사도를 기반으로 관광지를 추천하고 결과를 시각화합니다.

### generate_good_to_know.py
관광지의 부정 리뷰 측면을 바탕으로 GPT를 이용해 방문 전 참고사항(Good to Know)을 생성하는 코드입니다.

### gpt_41_mini.py
GPT-4.1 mini를 이용하여 TripAdvisor 리뷰에서 Aspect-Based Sentiment Analysis(ABSA)를 수행하고, 측면(aspect)과 감성(sentiment)을 추출합니다.

### normalize_aspects.py
추출된 다양한 aspect 표현을 대표 카테고리(canonical aspect)로 정규화하는 코드입니다.

### tourist_profile_data.csv
관광지 프로필 데이터입니다. 관광지 소개(Intro), 대표 장점, Good to Know, 평균 평점 등 추천 시스템에서 사용하는 정보를 포함합니다.

### images/
각 관광지의 이미지를 저장하는 폴더입니다. 추천 결과 화면에서 사용됩니다.

### .streamlit/
Streamlit 실행에 필요한 설정 파일을 저장하는 폴더입니다.

### requirements.txt
프로젝트 실행에 필요한 Python 라이브러리 목록입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```