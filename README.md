# BIM/BM25 기반 한국어 검색 시스템 (Korean Search Engine Project)
컴퓨터공학과 2170045 서자영

---

## [최종보고서 버전 readme]

### [최종 보고서 바로가기](reports/Final_Report.ipynb)

### 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 정보 검색(Information Retrieval)의 대표적인 확률 모델인 **BIM**과 **BM25**를 직접 구현하고, 한국어 위키 데이터셋(**KomuRetrieval**)을 활용하여 두 모델의 성능 차이를 비교 분석한다.
특히, 검색 성능에 영향을 미치는 주요 변수들을 **Multinomial Linear Regression Model**로 정의하고, 각 변수에 대한 통계적 분석을 수행했다.

#### 연구 목표
1.  **모델 구현:** BIM과 BM25 알고리즘의 직접 구현 (Python)
2.  **성능 비교:** MAP, P@10, R@10 지표를 통한 정량적 성능 평가
3.  **통계 분석:** `y = a1*x1 + a2*x2 + a3*x3 + b` 형태의 **Multinomial Linear Regression Model** 구축 및 분석
    * `x1` (**a1**): 문서 길이 (Document Length)
    * `x2` (**a2**): 쿼리 길이 (Query Length)
    * `x3` (**a3**): 도메인/토픽 (Dominant Topic by LDA)

---

### 2. 최종 프로젝트 구조 (Final Directory Structure)

최종 보고서 및 분석 코드는 아래 디렉토리에 정리되어 있다.

```text
NLP/
├── final_notebooks/              # [최종] 소스 코드 (순서대로 실행)
│   ├── 1_Preprocessing.ipynb           # 전처리, 불용어 처리, LDA 토픽 모델링
│   ├── 2_Retrieval_and_Metrics.ipynb   # 역색인 구축, 검색 수행, 하이퍼파라미터 튜닝
│   └── 3_Regression_and_Visualization.ipynb # 회귀 분석, 시각화
│
├── data_final/                   # [최종] 분석 결과 데이터 및 시각화
│   ├── sampled_data_v2.pkl             # 전처리 완료된 데이터셋 (Smart Sampling)
│   ├── tuning_fine_v2.csv              # BM25 파라미터 튜닝 결과
│   ├── performance_metrics_v2.csv      # 모델 성능 지표
│   ├── odds_ratio_v2.csv               # 회귀 분석 결과
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
3.  **`3_Regression_and_Visualization.ipynb`**: 통계적 가설 검증(회귀 분석), Odds Ratio 계산, 결과 시각화

-----

### 4\. 주요 실험 결과 (Key Results)

#### 1\. 검색 성능 비교

BM25가 BIM 대비 모든 지표에서 성능 향상을 보였다.

| 평가 지표 | BIM | BM25 (Best) | 개선율 |
| :--- | :--- | :--- | :--- |
| **MAP** | 0.416 | **0.631** | **+51.5%** |
| **P@10** | 0.193 | **0.270** | **+40.0%** |
| **R@10** | 0.520 | **0.681** | **+31.0%** |

#### 2\. Multinomial Linear Regression 분석 (y = a1·x1 + a2·x2 + a3·x3 + b)

검색 성공 여부(Relevance)를 종속변수로 하여 회귀 분석을 수행한 결과, 각 변수의 영향력과 그 원인은 다음과 같이 분석되었다.

  * **a1: 문서 길이 (Document Length)**

      * **BIM (Odds Ratio 0.89):** 통계적으로 유의한 짧은 문서 편향(Short Document Bias)이 확인되었다. BIM은 단어의 빈도(Frequency)를 고려하지 않고 단순히 유/무(Binary)만 따지기 때문에, 길이가 긴 문서에서 발생하는 '정보의 희석'이나 '노이즈'를 효과적으로 제어하지 못해 상대적으로 짧은 문서를 선호하는 경향을 보였다.
      * **BM25 (Odds Ratio 1.01):** Odds Ratio가 1에 가까워 문서 길이가 검색 성공에 영향을 미치지 않는 중립적 상태임이 입증되었다. 이는 하이퍼파라미터 튜닝을 통해 설정한 길이 정규화 상수 `b=0.99`가 긴 문서에 대한 페널티를 적절히 부과하여, 문서 길이의 유불리를 완벽하게 제거했음을 의미한다.

  * **a2: 쿼리 길이 (Query Length)**

      * **공통 결과:** 두 모델 모두 Odds Ratio가 1보다 크게 나타났다 (BIM 1.07, BM25 1.03).
      * **해석:** 사용자가 입력하는 쿼리의 길이가 길어질수록(형태소 개수가 많을수록), 검색 의도가 구체적이고 명확해지기 때문에 검색 성공 확률이 높아진다. 특히 BIM이 더 높은 수치를 보인 이유는, 빈도 정보 없이 단어 매칭에만 의존하는 모델 특성상 쿼리 텀이 많을수록 매칭 확률이 높아지는 것에 더 민감하게 반응하기 때문이다.

  * **a3: 도메인/토픽 (Dominant Topic)**

      * **결과:** 가중치가 **0.0**으로 수렴하며 유의미한 영향력이 없음이 확인되었다.
      * **해석 (변별력 부재):** LDA 토픽 모델링은 문서를 '스포츠', '역사'와 같은 거시적 주제(Macro Topic)로 분류한다. 하지만 실제 검색 환경에서는 정답 문서와 오답 문서가 동일한 주제(예: 둘 다 '역사' 문서)를 공유하는 경우가 대부분이다. 따라서 토픽 정보는 정답과 오답을 구별하는 변별력(Discriminatory Power)을 갖지 못하며, 랭킹 요소로서 부적합하다는 결론을 도출했다.

  * **최적 파라미터:** `k1=3.25`, `b=0.99`

<br><br>

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
