# 🔍 BIM/BM25 기반 한국어 검색 시스템 (Korean Search Engine Project)

정보 검색(Information Retrieval)의 대표적인 확률 모델인 BIM(Binary Independence Model)과 BM25를 직접 구현하고, 한국어 데이터셋(**KomuRetrieval**)을 활용하여 두 모델의 성능 차이를 비교 분석하는 프로젝트입니다.

2170045 서자영

[📊 중간 보고서](https://github.com/xeoxaxeo/NLP/blob/main/reports/Intermediate_Report.ipynb)

-----

## 📂 프로젝트 구조 (Project Structure)

```text
NLP/
├── 📁 analysis/              # 실험 결과 그래프 및 시각화 자료
│   ├── basic_...             # 1차 검증(Basic Analysis) 결과
│   └── advanced_...          # 심층 분석(Deep Analysis) 결과 (길이, 복잡도, 쿼리 유형별)
├── 📁 data/                  # 데이터셋 (Pickle 파일)
│   ├── corpus.pkl            # 전체 문서 집합
│   ├── queries.pkl           # 검색 질의 집합
│   └── qrels.pkl             # 정답 데이터
├── 📁 database/              # 역색인(Inverted Index) DB 파일
│   ├── inverted_index.7z     # [필수] 실행 전 압축 해제 필요
│   └── ...                   # 버전별 DB (sample / full_dirty / clean)
├── 📁 notebooks/             # 주피터 노트북 (소스 코드)
│   ├── 00_Check_Data...      # 데이터 탐색(EDA) 및 진단
│   ├── 01_Indexing...        # 데이터 전처리 및 색인 구축
│   ├── 02_Search...          # 검색 수행 및 결과 저장
│   └── 03_Analysis...        # 성능 평가 및 시각화
├── 📁 reports/               # 보고서 파일
│   └── Intermediate_Report.ipynb  # 중간 보고서
└── 📁 results/               # 검색 결과 JSON 파일
    └── search_results...
```

-----

## 🚀 실행 방법 (Getting Started)

### ⚠️ 중요: 실행 전 데이터베이스 준비

> **GitHub 용량 제한으로 인해 대용량 DB 파일들은 `.7z`로 분할 압축하여 업로드했습니다.**
> **코드 실행 전 압축을 해제하고, `.db` 파일을 생성해야 합니다.**

### 🛠️ 환경 설정 (Requirements)

  * **Python Version:** Python 3.14
  * **Library:** 아래 명령어로 필수 라이브러리를 설치하세요.

<!-- end list -->

```bash
pip install jupyter ipykernel kiwipiepy datasets tqdm pandas matplotlib seaborn scikit-learn kss streamlit
```

*(또는 `requirements.txt` 사용)*

-----

## 📄 중간보고서 관련 파일 상세 설명

### 📁 Report
#### /report
  * `Intermediate_Report.ipynb`: 중간보고서

### 📁 Data & Database
#### /data
  * `*.pkl`: 원본 데이터셋 (Corpus, Queries, Qrels)

#### /database
  * `inverted_index.7z`: 원본 데이터셋 DB
  * `inverted_index_sample10000.7z`: 1만 개 샘플링 데이터셋 DB

### 📁 Source Codes
#### /notebooks
  * `01_Indexing_Full_Dirty.ipynb`: 원본 데이터셋 인덱싱
  * `01_Indexing_Sample.ipynb`: 1만 개 샘플 인덱싱

  * `02_Search_Full_Dirty.ipynb`: 원본 데이터셋 기반 검색 수행
  * `02_Search_Sample.ipynb`: 1만 개 샘플 기반 검색 수행

  * `03_Basic_Analysis_Full_Dirty.ipynb`: 원본 데이터셋 사용 모델 성능 분석
  * `03_Basic_Analysis_Sample10000.ipynb`: 1만 개 샘플 사용 모델 성능 분석

### 📁 Results & Analysis

#### /results

  * `search_results.json`: 원본 데이터셋 사용 모델 검색 결과
  * `search_results_sample10000.json`: 1만 개 샘플 사용 모델 검색 결과

#### /analysis

  * `basic_analysis_full.png`: 원본 데이터셋 사용 모델 성능 그래프
  * `basic_analysis_sample.png`: 1만 개 샘플 사용 모델 샘플 성능 그래프
