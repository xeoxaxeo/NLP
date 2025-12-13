# BIM/BM25 기반 한국어 검색 시스템 구현 및 모델 비교 분석

**컴퓨터공학과 2170045 서자영**

[최종보고서 바로가기](reports/FINAL_REPORT_V2.ipynb)

---

## 1. 프로젝트 개요

BIM과 BM25 모델을 한국어 데이터셋(KomuRetrieval)에 적용하여 검색 성능을 비교하고, **3가지 회귀 분석**(Binary Logistic Regression, Multinomial Logistic Regression, Linear Multinomial Regression) 을 통해 변수별 영향력을 통계적으로 검증한다.

### 핵심 결과

| 지표 | BIM | BM25 | 개선율 |
|------|-----|------|--------|
| MAP | 0.416 | 0.631 | **+51.5%** |
| P@10 | 0.193 | 0.270 | **+40.0%** |
| R@10 | 0.520 | 0.681 | **+31.0%** |

**통계적 발견:**
- BIM: 짧은 문서 편향 (Binary OR=0.89, Linear a1=-0.026)
- BM25: 공정성 (Binary OR=1.01, Linear a1=+0.059)
- Long 문서: BM25 MAP 0.824 vs BIM 0.593 (+39%)
- 최적 파라미터: k1=3.25, b=0.99

---

## 2. 최종보고서 관련 파일 구조

```
NLP/
├── final_notebooks/
│   ├── 1_data_preprocessing.ipynb      # 전처리, LDA
│   ├── 2_indexing_retrieval.ipynb      # BM25 튜닝, 검색
│   ├── 3_logistic_regression.ipynb     # Binary Logistic
│   ├── 4_subgroup_analysis.ipynb       # 세부 그룹 분석
│   └── 5_linear_regression.ipynb       # Linear Regression
│
├── data_final/                          # 결과 데이터 및 시각화
│   ├── sampled_data_v2.7z              # 전처리 데이터 (압축 해제 필요)
│   ├── regression_dataset_v2.csv       # 27,298개 회귀 데이터
│   ├── odds_ratio_v2.csv               # Binary Logistic 결과
│   ├── linear_regression_*_v5.csv      # Linear Regression 결과
│   ├── *_performance_v3.csv            # 세부 그룹별 성능
│   └── *.png                           # 시각화
│
├── database_final/
│   └── search_index_v2.7z              # SQLite DB (압축 해제 필요)
│
├── data/                                # 원본 데이터
│   ├── corpus.pkl                      # 50,222개 문서
│   ├── queries.pkl                     # 1,454개 쿼리
│   └── qrels.pkl                       # 관련성 레이블
│
└── reports/
    └── FINAL_REPORT_V2.ipynb           # 최종보고서
```

---

## 3. 환경 설정

### 필수 라이브러리

```bash
pip install jupyter pandas numpy matplotlib seaborn
pip install kiwipiepy gensim scikit-learn statsmodels
```

### 압축 해제

실행 전 data_final, database_final, database, data 디렉토리의 압축된 파일들 압축 해제 필수

---

## 4. 프로젝트 실행 로직

### Step 1: 전처리 (`1_data_preprocessing.ipynb`)
```
입력: corpus.pkl (50,222개)
처리:
  - Smart Sampling (5,000개, 정답 문서 우선)
  - Kiwi 형태소 분석
  - IDF 기반 불용어 추출 (346개)
  - 'O' 패딩 처리 (길이 보존)
  - LDA 토픽 모델링 (10개)
출력: sampled_data_v2.pkl, stopwords_v2.txt 등
```

### Step 2: 검색 시스템 (`2_indexing_retrieval.ipynb`)
```
입력: sampled_data_v2.pkl
처리:
  - SQLite 역색인 구축
  - BM25 하이퍼파라미터 튜닝
    ├─ Coarse: k1=[2.0~5.0], b=[0.6~0.99]
    └─ Fine: 최적값 주변 세밀 탐색
  - BIM/BM25 검색 수행
출력: search_index_v2.db, regression_dataset_v2.csv 등
```

### Step 3: Binary Logistic (`3_logistic_regression.ipynb`)
```
입력: regression_dataset_v2.csv (27,298개)
처리:
  - VIF 진단 (다중공선성)
  - Binary Logistic Regression
  - Odds Ratio 계산
  - 가중치 튜닝 (270개 조합)
출력: odds_ratio_v2.csv, weight_tuning_v2.csv 등
```

### Step 4: 세부 분석 (`4_subgroup_analysis.ipynb`)
```
입력: regression_dataset_v2.csv
처리:
  - 문서 길이별 (Short/Medium/Long)
  - 토픽별 (Topic 0~9)
  - 쿼리 복잡도별 (Simple/Medium/Complex)
  - Multinomial Logistic Regression
출력: *_performance_v3.csv, *_odds_ratio_v3.csv 등
```

### Step 5: Linear Regression (`5_linear_regression.ipynb`)
```
입력: regression_dataset_v2.csv
처리:
  - 토픽 더미 변수 변환
  - Linear Regression (y = 검색 점수)
  - 통계적 유의성 검증 (p-value, CI, t-stat)
  - 토픽별 계수 분석
출력: linear_regression_*_v5.csv 등
```

---

## 5. 프로젝트 결과 요약

### 5.1 전체 성능

| 모델 | MAP | P@10 | R@10 |
|------|-----|------|------|
| BIM | 0.416 | 0.193 | 0.520 |
| BM25 | 0.631 | 0.270 | 0.681 |

### 5.2 통계 분석 결과

**Binary Logistic Regression:**
- BIM: Pseudo R²=0.234, Doc Length OR=0.89 *** (짧은 문서 편향)
- BM25: Pseudo R²=0.016, Doc Length OR=1.01 *** (중립)

**Linear Multinomial Regression:**
- BIM: a1=-0.026 ***, a2=+0.689 ***, Test R²=0.501 (편향)
- BM25: a1=+0.059 **, a2=+1.322 ***, Test R²=0.262 (공정)
- **a3 토픽 계수 (BIM)**: 6개 유의미 (Topic 2, 3, 6, 7, 8, 9)
- **a3 토픽 계수 (BM25)**: 7개 유의미 (Topic 2, 3, 4, 6, 7, 8, 9)
- Topic 7 (마인크래프트): BIM +1.66 ***, BM25 +6.90 *** (최고 영향력)

**Linear Regression 상세 계수:**

| 변수 | BIM | BM25 | 해석 |
|------|-----|------|------|
| **a1 (doc)** | -0.026 *** | +0.059 ** | BIM: 1k자 증가 시 -0.026점 |
| **a2 (query)** | +0.689 *** | +1.322 *** | BM25가 2배 민감 |
| **a3_2 (스포츠)** | +1.69 *** | +5.31 *** | 유의미 |
| **a3_3 (사회)** | +1.09 *** | +2.68 *** | 유의미 |
| **a3_6 (전쟁)** | +1.07 *** | +2.69 *** | 유의미 |
| **a3_7 (마크)** | +1.66 *** | +6.90 *** | 최고 |
| **a3_8 (방송)** | +0.69 *** | +2.68 *** | 유의미 |
| **a3_9 (교통)** | +0.59 * | +3.08 *** | 유의미 |

### 5.3 세부 그룹별 성능 (MAP)

| 그룹 | BIM | BM25 | 차이 |
|------|-----|------|------|
| Long 문서 | 0.593 | 0.824 | +0.231 |
| Complex 쿼리 | 0.514 | 0.764 | +0.250 |

### 5.4 BM25 최적 파라미터

- k1 = 3.25 (TF 포화 계수)
- b = 0.99 (길이 정규화 강도)

---

## 6. 데이터셋

### KomuRetrieval (나무위키 기반)

| 항목 | 수량 | 설명 |
|------|------|------|
| **원본 문서** | 50,222개 | 전체 Corpus |
| **샘플 문서** | 5,000개 | Smart Sampling |
| **쿼리** | 1,454개 | 검색 질의 |
| **회귀 데이터** | 27,298개 | Query-Document 쌍 |

### 문서 특성

- 평균 길이: 7,873자 (중앙값 4,366자)
- 길이 범위: 0 ~ 102,419자 (편차 매우 큼)
- LDA 토픽: 10개 (역사/정치, 일상/감정, 스포츠, 게임, 등)

---

## 전체 파일 구조
```
NLP/
├── final_notebooks/              # [최종] 소스 코드 (순서대로 실행)
│   ├── 1_data_preprocessing.ipynb       # 전처리, 불용어 OOO 패딩, LDA 토픽 모델링
│   ├── 2_indexing_retrieval.ipynb       # 역색인, BM25 튜닝 (k1=3.25, b=0.99)
│   ├── 3_logistic_regression.ipynb      # Binary Logistic, Odds Ratio
│   ├── 4_subgroup_analysis.ipynb        # Multinomial Logistic (세부 그룹)
│   └── 5_linear_regression.ipynb        # Linear Multinomial Regression
│
├── data_final/                   # [최종] 분석 결과 데이터 및 시각화
│   ├── sampled_data_v2.7z               # 전처리 완료 (5,000개, Smart Sampling)
│   ├── stopwords_v2.txt                 # IDF < 1.5 불용어 346개
│   ├── regression_dataset_v2.csv        # 27,298개 회귀 데이터
│   │
│   ├── lda_model_v2.model*              # LDA 토픽 모델 파일들
│   │
│   ├── tuning_coarse_v2.csv             # BM25 1차 튜닝 결과
│   ├── tuning_fine_v2.csv               # BM25 2차 튜닝 결과 (최종)
│   ├── tuning_extended_final.csv        # 확장 튜닝 결과
│   ├── manual_tuning_results_final.csv  # 수동 튜닝 결과
│   │
│   ├── performance_metrics_v2.csv       # MAP/P@10/R@10
│   ├── vif_full_v2.csv                  # 다중공선성 진단 (전체)
│   ├── vif_final_v2.csv                 # 다중공선성 진단 (최종)
│   │
│   ├── odds_ratio_v2.csv                # Binary Logistic 결과
│   ├── weight_tuning_v2.csv             # 가중치 튜닝 결과
│   ├── summary_metrics_v2.csv           # 모델 성능 요약
│   │
│   ├── linear_regression_results_v5.csv # Linear Regression 계수
│   ├── linear_regression_stats_v5.csv   # 통계적 유의성 (p-value, CI)
│   ├── linear_regression_topic_coefs_v5.csv  # 토픽별 계수
│   │
│   ├── length_analysis_v2.csv           # 문서 길이별 분석 원본
│   ├── length_performance_v3.csv        # 문서 길이별 성능
│   ├── length_odds_ratio_v3.csv         # 길이별 Odds Ratio
│   │
│   ├── topic_performance_v3.csv         # 토픽별 성능
│   │
│   ├── complexity_analysis_v2.csv       # 쿼리 복잡도별 분석 원본
│   ├── complexity_performance_v3.csv    # 쿼리 복잡도별 성능
│   ├── complexity_odds_ratio_v3.csv     # 복잡도별 Odds Ratio
│   │
│   └── *.png                            # 시각화 이미지
│       ├── odds_ratio_comparison_v2.png
│       ├── performance_comparison_v2.png
│       ├── weight_heatmap_v2.png
│       ├── linear_regression_v5.png
│       ├── length_map_trend_v3.png
│       ├── topic_map_trend_v3.png
│       ├── complexity_map_trend_v3.png
│       └── additional_analysis_v3.png
│
├── database_final/               # [최종] 검색 엔진 DB
│   └── search_index_v2.7z               # SQLite 역색인 (5,000개 문서)
│                                        # ⚠️ 압축 해제 필요
│
├── database/                     # [중간] 검색 엔진 DB (중간보고서용)
│
├── notebooks/                    # [중간] 소스 코드 (중간보고서용)
│
├── analysis/                     # [중간] 시각화 결과 (중간보고서용)
│
├── results/                      # [중간] .json (중간보고서용)
│
├── data/                         # 원본 데이터셋
│   ├── corpus.pkl                       # 전체 50,222개 문서
│   ├── queries.pkl                      # 1,454개 쿼리
│   └── qrels.pkl                        # Query-Document 관련성 레이블
│
└── reports/                      # 보고서
    ├── FINAL_REPORT_V2.ipynb            # [최종 보고서] ⭐
    └── Intermediate_Report.ipynb        # [중간 보고서]
```
