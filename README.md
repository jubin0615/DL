# Deep Learning & Data Analysis

딥러닝 모델 구현, 위성 데이터 분석 코드

## 📂 프로젝트 구성 (Project Structure)

### 1. 딥러닝 기초 (`DL_HW1.ipynb`)
NumPy만을 사용하여 신경망(Neural Network)을 바닥부터(from scratch) 구현하고, 텍스트 데이터에 대한 이진 분류를 수행한다.

- **주요 내용:**
  - 텍스트 전처리 (소문자 변환, 특수문자 제거 등)
  - **SimpleNN 구현:** Embedding Layer와 Fully Connected Layer를 포함한 신경망 클래스 직접 구현
  - **역전파(Backpropagation):** 자동 미분 라이브러리 없이 `forward`, `backward` 패스 및 손실 함수(Binary Cross-Entropy) 직접 계산
  - **데이터:** 영화 리뷰 데이터 (`hw1_data.tsv`)

### 2. BERT를 활용한 텍스트 분류 (`HW2_1_Classification.ipynb`)
사전 학습된(Pre-trained) **BERT** 모델을 사용하여 CoLA(Corpus of Linguistic Acceptability) 데이터셋을 분류한다.

- **주요 내용:**
  - **Task:** Binary Classification
  - **Dataset:** GLUE 벤치마크의 CoLA 데이터셋
  - **Model:** `bert-base-uncased` (Hugging Face Transformers)
  - **Training:** PyTorch 학습 루프 구현 및 AdamW 옵티마이저 활용
  - **Metric:** Accuracy

### 3. mBART를 활용한 기계 번역 (`HW2_2_Generation.ipynb`)
다국어 번역 모델인 **mBART**를 파인튜닝하여 프랑스어(French)를 영어(English)로 번역하는 생성 모델 프로젝트

- **주요 내용:**
  - **Task:** Machine Translation (French → English)
  - **Dataset:** IWSLT 2017
  - **Model:** `facebook/mbart-large-50-many-to-many-mmt`
  - **Training:** Hugging Face `Seq2SeqTrainer`를 활용한 Fine-tuning
  - **Metric:** SacreBLEU 점수를 통한 성능 평가

### 4. ISS 궤도 데이터 분석 (`TLE_ISS.ipynb`)
국제우주정거장(ISS)의 궤도 데이터(Orbit Data)를 시각화하고 분석

- **주요 내용:**
  - **데이터:** ISS Orbit Data (`Orbit-Data-25544-GP.csv`)
  - **분석 항목 (Visualization):**
    - **Orbit Decay:** 시간 경과에 따른 궤도 장반경(Semi-Major Axis) 감소 추이
    - **Inclination:** 궤도 경사각의 변화
    - **Eccentricity:** 궤도 이심률 변화

## 🛠️ 사용 라이브러리 (Dependencies)

이 프로젝트들을 실행하기 위해 다음의 라이브러리들이 필요합니다.

- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `sacrebleu`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## 🚀 실행 방법 (Usage)

각 노트북 파일(`.ipynb`)은 Google Colab 또는 로컬 Jupyter Notebook 환경에서 실행할 수 있다.
