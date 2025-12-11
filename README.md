# BIM/BM25 기반 한국어 검색 시스템 (Korean Search Engine Project)
컴퓨터공학과 2170045 서자영

---

## [최종보고서 버전 readme]

### [최종 보고서 바로가기](reports/Final_Report.ipynb)

### 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 정보 검색(Information Retrieval)의 대표적인 확률 모델인 **BIM**과 **BM25**를 직접 구현하고, 한국어 위키 데이터셋(**KomuRetrieval**)을 활용하여 두 모델의 성능 차이를 비교 분석한다. 검색 성능에 영향을 미치는 주요 변수들을 통계적으로 분석하기 위해 이진 회귀 분석, 다항 회귀 분석을 실시한다.

#### 연구 목표
1.  **모델 구현:** BIM과 BM25 알고리즘의 직접 구현 (Python)
2.  **심층 분석**: 전체 성능뿐만 아니라 **세부 그룹(Sub-group)**별 성능 차이 검증
    * Long Documents (장문 문서에서의 강건성)
    * Complex Queries (복잡한 질의 처리 능력)
    * Topic Dependency (특정 주제 편향 여부)
3.  **통계적 검증**: 회귀 모델 $y = a_1x_1 + a_2x_2 + a_3x_3 + b$ 수립 및 Odds Ratio 분석
   
---

### 2. 최종 프로젝트 구조 (Final Directory Structure)

최종 보고서 및 분석 코드는 아래 디렉토리에 정리되어 있다.

```text
NLP/
├── final_notebooks/              # [최종] 소스 코드 (순서대로 실행)
│   ├── 1_Preprocessing.ipynb           # 전처리, 불용어 처리, LDA 토픽 모델링
│   ├── 2_Retrieval_and_Metrics.ipynb   # 역색인 구축, 검색 수행, 하이퍼파라미터 튜닝
│   ├── 3_Regression_and_Visualization.ipynb  # 이진 회귀 분석, 가중치 튜닝
│   └── 4_Advanced_Analysis.ipynb             # 다항 회귀 분석
│
├── data_final/                   # [최종] 분석 결과 데이터 및 시각화
│   ├── sampled_data_v2.pkl             # 전처리 완료된 데이터셋 (Smart Sampling)
│   ├── tuning_fine_v2.csv              # BM25 파라미터 튜닝 결과
│   ├── performance_metrics_v2.csv      # 모델 성능 지표
│   ├── odds_ratio_v2.csv               # 회귀 분석 결과
│   ├── length_performance_v3.csv        # 문서 길이별 성능 데이터
│   ├── topic_performance_v3.csv         # 토픽별 성능 데이터
│   ├── complexity_performance_v3.csv    # 쿼리 복잡도별 성능 데이터
│   ├── *.png                           # 결과 그래프 (Heatmap, Performance, etc.)
│   └── ...
│
├── database_final/               # [최종] 검색 엔진 DB
│   └── search_index_final.db           # SQLite 기반 역색인 파일
│
├── reports/                      # 보고서 파일
│   ├── Final_Report.ipynb              # [최종 보고서]
│   └── Intermediate_Report.ipynb       # [중간 보고서]
│
└── data/                         # 원본 데이터셋 (Corpus, Queries, Qrels)
````

-----

### 3\. 환경 설정 및 실행 (Setup & Usage)

#### 중요: 대용량 데이터베이스 파일 처리

GitHub 용량 제한으로 인해 데이터베이스 파일들은 압축(.7z)하여 업로드했다.
코드 실행 전 반드시 data/, data_final/, database_final/ 폴더 내의 .7z 파일 압축을 해제해야 정상적으로 작동한다.

#### 필수 라이브러리 설치

본 프로젝트는 **Python 3.11.9** 환경으로 구축했다. 아래 명령어로 필요한 라이브러리를 설치한다.

```bash
pip install jupyter ipykernel kiwipiepy datasets tqdm pandas numpy matplotlib seaborn scikit-learn kss streamlit
```

#### 실행 순서

`final_notebooks` 폴더 내의 노트북 파일을 번호 순서대로 실행하면 전체 실험 과정을 재현할 수 있다.

1.  **`1_Preprocessing.ipynb`**: 데이터 로드, Kiwi 형태소 분석, 불용어(IDF \< 1.5) 처리, LDA 토픽 모델링 수행
2.  **`2_Retrieval_and_Metrics.ipynb`**: SQLite 역색인 구축, BM25 파라미터 튜닝(Coarse/Fine Grid), 검색 수행
3.  **`3_Regression_and_Visualization.ipynb`**: 통계적 가설 검증(이진 회귀 분석), Odds Ratio 계산, 결과 시각화, 가중치 튜닝
4.  **`4_Advanced_Analysis.ipynb`**: 통계적 가설 검증(다항 회귀 분석), Odds Ratio 계산, 결과 시각화
   
-----

## [중간보고서 버전 readme]

### [중간 보고서 바로가기](https://www.google.com/search?q=reports/Intermediate_Report.ipynb)

### 프로젝트 구조 (Intermediate Structure)

```text
NLP/
├── analysis/                 # 실험 결과 그래프 및 시각화 자료
│   ├── basic_...             # 1차 검증(Basic Analysis) 결과
│   └── advanced_...          # 심층 분석(Deep Analysis) 결과 (길이, 복잡도, 쿼리 유형별)
├── data/                     # 데이터셋 (Pickle 파일)
│   ├── corpus.pkl            # 전체 문서 집합
│   ├── queries.pkl           # 검색 질의 집합
│   └── qrels.pkl             # 정답 데이터
├── database/                 # 역색인(Inverted Index) DB 파일
│   ├── inverted_index.7z     # [필수] 실행 전 압축 해제 필요
│   └── ...                   # 버전별 DB (sample / full_dirty / clean)
├── notebooks/                # 주피터 노트북 (소스 코드)
│   ├── 00_Check_Data...      # 데이터 탐색(EDA) 및 진단
│   ├── 01_Indexing...        # 데이터 전처리 및 색인 구축
│   ├── 02_Search...          # 검색 수행 및 결과 저장
│   └── 03_Analysis...        # 성능 평가 및 시각화
├── reports/                  # 보고서 파일
│   └── Intermediate_Report.ipynb  # 중간 보고서
└── results/                  # 검색 결과 JSON 파일
    └── search_results...
```

-----

### 중간보고서 관련 파일 상세 설명

#### Report

  * `reports/Intermediate_Report.ipynb`: [중간 보고서 바로가기](https://www.google.com/search?q=reports/Intermediate_Report.ipynb)

#### Data & Database

  * `/data/*.pkl`: 원본 데이터셋 (Corpus, Queries, Qrels)
  * `/database/inverted_index.7z`: 원본 데이터셋 DB
  * `/database/inverted_index_sample10000.7z`: 1만 개 샘플링 데이터셋 DB

#### Source Codes (/notebooks)

  * `01_Indexing_Full_Dirty.ipynb`: 원본 데이터셋 인덱싱
  * `01_Indexing_Sample.ipynb`: 1만 개 샘플 인덱싱
  * `02_Search_Full_Dirty.ipynb`: 원본 데이터셋 기반 검색 수행
  * `02_Search_Sample.ipynb`: 1만 개 샘플 기반 검색 수행
  * `03_Basic_Analysis_Full_Dirty.ipynb`: 원본 데이터셋 사용 모델 성능 분석
  * `03_Basic_Analysis_Sample10000.ipynb`: 1만 개 샘플 사용 모델 성능 분석

#### Results & Analysis

  * `/results/search_results.json`: 원본 데이터셋 사용 모델 검색 결과
  * `/results/search_results_sample10000.json`: 1만 개 샘플 사용 모델 검색 결과
  * `/analysis/basic_analysis_full.png`: 원본 데이터셋 사용 모델 성능 그래프
  * `/analysis/basic_analysis_sample.png`: 1만 개 샘플 사용 모델 샘플 성능 그래프
