# setup.py: 환경 설정 및 데이터셋 로드 (전체 데이터)

import os
import sys
import pickle
from typing import List, Tuple
from datetime import datetime
from datasets import load_dataset
from kiwipiepy import Kiwi
from tqdm import tqdm

DATA_DIR = 'data'
LOG_DIR = 'logs'

# 로그 파일 설정
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# 디렉토리 구조 생성
def setup_directories():
    directories = [
        'data',       # 데이터셋 저장
        'index',      # SQLite 인덱스 저장
        'results',    # 검색 결과 JSON 저장
        'analysis',   # 분석 결과 및 시각화
        'logs',       # 실행 로그 저장
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("\n디렉토리 구조 생성 완료\n")

# 전처리 함수 정의
def tokenize(text: str, kiwi_instance: Kiwi) -> List[str]:
    if not text or not text.strip():
        return []

    tokens = kiwi_instance.tokenize(text)
    useful_tags = ['NNG', 'NNP', 'VV', 'VA', 'MAG']

    result = []
    for token in tokens:
        if token.tag in useful_tags and len(token.form) > 1:
            result.append(token.form)

    return result

def test_tokenizer(kiwi: Kiwi):
    sample_text = "이화여자대학교는 1886년에 스크랜튼 여사가 설립하였다."
    sample_tokens = tokenize(sample_text, kiwi)

    print(f"\n  원문: {sample_text}")
    print(f"  토큰: {sample_tokens}")
    print("\n전처리 함수 정의 완료\n")

# KomuRetrieval 데이터셋 로드
def load_komu_dataset() -> Tuple[List, List, List]:
    print("=" * 60)
    print("전체 데이터셋 로드 시작")
    print("=" * 60 + "\n")

    queries = load_dataset("junyoungson/KomuRetrieval", "queries", split="queries")
    corpus = load_dataset("junyoungson/KomuRetrieval", "corpus", split="corpus")
    qrels = load_dataset("junyoungson/KomuRetrieval", split="test")

    print(f"\n원본 데이터셋 정보:")
    print(f"  - Corpus: {len(corpus):,}개 문서")
    print(f"  - Queries: {len(queries):,}개 쿼리")
    print(f"  - Qrels: {len(qrels):,}개 쿼리-문서 쌍")

    # 전체 데이터를 리스트로 변환
    print(f"\n전체 데이터 변환 시작")
    full_corpus = [corpus[i] for i in tqdm(range(len(corpus)), desc="Corpus 변환")]
    full_queries = [queries[i] for i in tqdm(range(len(queries)), desc="Queries 변환")]
    full_qrels = [qrels[i] for i in tqdm(range(len(qrels)), desc="Qrels 변환")]

    print(f"\n전체 데이터 로드 완료")
    print(f"  - Corpus: {len(full_corpus):,}개")
    print(f"  - Queries: {len(full_queries):,}개")
    print(f"  - Qrels: {len(full_qrels):,}개\n")

    return full_corpus, full_queries, full_qrels

# 데이터 샘플 출력
def show_data_samples(corpus: List, queries: List, qrels: List):
    print("\n[ Corpus 예시 ]")
    sample_doc = corpus[0]
    print(f"  ID: {sample_doc['_id']}")
    print(f"  제목: {sample_doc['title']}")
    print(f"  내용: {sample_doc['text'][:100]}...")

    print("\n[ Query 예시 ]")
    sample_query = queries[0]
    print(f"  ID: {sample_query['_id']}")
    print(f"  내용: {sample_query['text']}")

    print("\n[ Qrels 예시 ]")
    sample_qrel = qrels[0]
    print(f"  Query ID: {sample_qrel['query-id']}")
    print(f"  Corpus ID: {sample_qrel['corpus-id']}")
    print(f"  Score: {sample_qrel['score']}")

    print()

# 데이터 저장
def save_data(corpus: List, queries: List, qrels: List):
    os.makedirs(DATA_DIR, exist_ok=True)

    print("데이터 저장 시작")

    with open(os.path.join(DATA_DIR, 'corpus.pkl'), 'wb') as f:
        pickle.dump(corpus, f)
    print(f"  corpus.pkl 저장 ({len(corpus):,}개)")

    with open(os.path.join(DATA_DIR, 'queries.pkl'), 'wb') as f:
        pickle.dump(queries, f)
    print(f"  queries.pkl 저장 ({len(queries):,}개)")

    with open(os.path.join(DATA_DIR, 'qrels.pkl'), 'wb') as f:
        pickle.dump(qrels, f)
    print(f"  qrels.pkl 저장 ({len(qrels):,}개)")

    print("\n데이터 저장 완료\n")

# main
def main():
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'setup_{timestamp}.log')

    # 디렉토리 생성 (로그 디렉토리 포함)
    setup_directories()

    # 로거 시작
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("BIM/BM25 검색 시스템 - 데이터 준비")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # Kiwi 초기화
        print("Kiwi 형태소 분석기 초기화 시작")
        kiwi = Kiwi()
        print("초기화 완료\n")

        # tokenizer 테스트
        test_tokenizer(kiwi)

        # 전체 데이터셋 로드
        corpus, queries, qrels = load_komu_dataset()

        # 데이터 샘플 출력
        show_data_samples(corpus, queries, qrels)

        # 데이터 저장
        save_data(corpus, queries, qrels)

        print("=" * 60)
        print("데이터 준비 완료")
        print(f"로그 파일: {log_file}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # 로거 종료
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n로그 저장: {log_file}")

if __name__ == "__main__":
    main()