# setup.py: 환경 설정 및 데이터셋 로드

import os
import random
import pickle
from typing import List, Tuple
from datasets import load_dataset
from kiwipiepy import Kiwi
from tqdm import tqdm

DATA_DIR = 'data'
SAMPLE_SIZE = 5000
RANDOM_SEED = 42

# 디렉토리 구조 생성
def setup_directories():

    directories = [
        'data',       # 데이터셋 저장
        'index',      # SQLite 인덱스 저장
        'results',    # 검색 결과 JSON 저장
        'analysis',   # 분석 결과 및 시각화
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("\n디렉토리 구조 생성 완료\n")

# 전처리 함수 정의
def tokenize(text: str, kiwi_instance: Kiwi) -> List[str]:

    if not text or not text.strip():
        return []

    # 형태소 분석
    tokens = kiwi_instance.tokenize(text)

    # 명사, 동사, 형용사만 추출
    useful_tags = ['NNG', 'NNP', 'VV', 'VA', 'MAG']

    result = []
    for token in tokens:
        if token.tag in useful_tags and len(token.form) > 1:  # 1글자 제외
            result.append(token.form)

    return result


def test_tokenizer(kiwi: Kiwi):

    sample_text = "이화여자대학교는 1886년에 스크랜튼 여사가 설립하였다."
    sample_tokens = tokenize(sample_text, kiwi)

    print(f"\n  원문: {sample_text}")
    print(f"  토큰: {sample_tokens}")
    print("\n전처리 함수 정의 완료!\n")


# KomuRetrieval 데이터셋 로드
def load_komu_dataset(sample_size: int = SAMPLE_SIZE) -> Tuple[List, List, List]:

    queries = load_dataset("junyoungson/KomuRetrieval", "queries", split="queries")
    corpus = load_dataset("junyoungson/KomuRetrieval", "corpus", split="corpus")
    qrels = load_dataset("junyoungson/KomuRetrieval", split="test")

    print(f"\n데이터셋 정보:")
    print(f"  - Corpus: {len(corpus):,}개 문서")
    print(f"  - Queries: {len(queries):,}개 쿼리")
    print(f"  - Qrels: {len(qrels):,}개 쿼리-문서 쌍")

    # Corpus 샘플링
    print(f"\n샘플링")
    random.seed(RANDOM_SEED)
    sample_size = min(sample_size, len(corpus))
    sampled_indices = random.sample(range(len(corpus)), sample_size)

    # 샘플링된 corpus 저장
    sampled_corpus = [corpus[i] for i in tqdm(sampled_indices, desc="Corpus Sampling")]

    # 샘플링된 문서 ID 집합
    sampled_doc_ids = {doc['_id'] for doc in sampled_corpus}

    # Qrels 필터링 (샘플링된 문서와 관련된 것만)
    filtered_qrels = []
    for qrel in tqdm(qrels, desc="Qrels Filtering"):
        if qrel['corpus-id'] in sampled_doc_ids:
            filtered_qrels.append(qrel)

    # 필터링된 쿼리 ID 집합
    filtered_query_ids = {qrel['query-id'] for qrel in filtered_qrels}

    # Queries 필터링
    filtered_queries = []
    for query in tqdm(queries, desc="Queries Filtering"):
        if query['_id'] in filtered_query_ids:
            filtered_queries.append(query)

    print(f"\n샘플링 완료")
    print(f"  - Corpus: {len(sampled_corpus):,}개 (전체 {len(corpus):,}개 중)")
    print(f"  - Queries: {len(filtered_queries):,}개 (전체 {len(queries):,}개 중)")
    print(f"  - Qrels: {len(filtered_qrels):,}개 (전체 {len(qrels):,}개 중)\n")

    return sampled_corpus, filtered_queries, filtered_qrels


# 데이터 샘플 출력
def show_data_samples(corpus: List, queries: List, qrels: List):
    # Corpus 샘플
    print("\n[ Corpus 예시 ]")
    sample_doc = corpus[0]
    print(f"  ID: {sample_doc['_id']}")
    print(f"  제목: {sample_doc['title']}")
    print(f"  내용: {sample_doc['text'][:100]}...")

    # Query 샘플
    print("\n[ Query 예시 ]")
    sample_query = queries[0]
    print(f"  ID: {sample_query['_id']}")
    print(f"  내용: {sample_query['text']}")

    # Qrels 샘플
    print("\n[ Qrels 예시 ]")
    sample_qrel = qrels[0]
    print(f"  Query ID: {sample_qrel['query-id']}")
    print(f"  Corpus ID: {sample_qrel['corpus-id']}")
    print(f"  Score: {sample_qrel['score']}")

    print()


# 데이터 저장
def save_data(corpus: List, queries: List, qrels: List):

    os.makedirs(DATA_DIR, exist_ok=True)

    # Corpus 저장
    with open(os.path.join(DATA_DIR, 'corpus.pkl'), 'wb') as f:
        pickle.dump(corpus, f)
    print(f"  corpus.pkl 저장 ({len(corpus):,}개)")

    # Queries 저장
    with open(os.path.join(DATA_DIR, 'queries.pkl'), 'wb') as f:
        pickle.dump(queries, f)
    print(f"  queries.pkl 저장 ({len(queries):,}개)")

    # Qrels 저장
    with open(os.path.join(DATA_DIR, 'qrels.pkl'), 'wb') as f:
        pickle.dump(qrels, f)
    print(f"  qrels.pkl 저장 ({len(qrels):,}개)")

    print("\n데이터 저장 완료\n")


# main

def main():

    # 디렉토리 생성
    setup_directories()

    # Kiwi 초기화
    kiwi = Kiwi()

    # tokenizer 테스트
    test_tokenizer(kiwi)

    # 데이터셋 로드 및 샘플링
    corpus, queries, qrels = load_komu_dataset(sample_size=SAMPLE_SIZE)

    # 데이터 샘플 출력
    show_data_samples(corpus, queries, qrels)

    # 데이터 저장
    save_data(corpus, queries, qrels)

if __name__ == "__main__":
    main()