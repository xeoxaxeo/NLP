# search.py: BIM/BM25 검색 엔진 구현

import os
import sqlite3
import json
import pickle
import math
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from kiwipiepy import Kiwi
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / 'data'
INDEX_DIR = PROJECT_ROOT / 'index'
RESULTS_DIR = PROJECT_ROOT / 'results'
DB_PATH = INDEX_DIR / 'inverted_index.db'

# BM25 파라미터
K1 = 1.2
B = 0.75

# 결과 반환 개수
TOP_K = 100

# 텍스트 토큰화
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

# 쿼리 데이터 로드
def load_queries():
    with open(os.path.join(DATA_DIR, 'queries.pkl'), 'rb') as f:
        queries = pickle.load(f)
    print(f"쿼리 로드 완료: {len(queries):,}개\n")
    return queries

# 통계 정보 조회
def get_statistics(cursor):
    cursor.execute("SELECT value FROM statistics WHERE key='N'")
    N = int(cursor.fetchone()[0])

    cursor.execute("SELECT value FROM statistics WHERE key='avgdl'")
    avgdl = float(cursor.fetchone()[0])

    return N, avgdl

# 특정 용어의 posting list 조회
def get_term_postings(cursor, term: str) -> Dict[str, int]:
    cursor.execute("""
        SELECT doc_id, tf 
        FROM inverted_index 
        WHERE term = ?
    """, (term,))

    postings = {}
    for doc_id, tf in cursor.fetchall():
        postings[doc_id] = tf

    return postings

# 문서 길이 조회
def get_document_length(cursor, doc_id: str) -> int:
    cursor.execute("SELECT length FROM documents WHERE doc_id = ?", (doc_id,))
    result = cursor.fetchone()
    return result[0] if result else 0

# 문서 정보 조회
def get_document_info(cursor, doc_id: str) -> Dict:
    cursor.execute("""
        SELECT doc_id, title, length 
        FROM documents 
        WHERE doc_id = ?
    """, (doc_id,))

    result = cursor.fetchone()
    if result:
        return {
            'doc_id': result[0],
            'title': result[1],
            'length': result[2]
        }
    return None

# IDF 계산
def calculate_idf(N: int, df: int) -> float:
    return math.log((N - df + 0.5) / (df + 0.5))

# BIM score 계산
def calculate_bim_score(query_tokens: List[str], cursor, N: int) -> Dict[str, float]:
    scores = defaultdict(float)

    for term in query_tokens:
        # Posting list 조회
        postings = get_term_postings(cursor, term)

        if not postings:
            continue

        # IDF 계산
        df = len(postings)
        idf = calculate_idf(N, df)

        # 해당 용어를 포함하는 모든 문서에 IDF 점수 부여
        for doc_id in postings.keys():
            scores[doc_id] += idf

    return scores

# BM25 score 계산
def calculate_bm25_score(query_tokens: List[str], cursor, N: int, avgdl: float) -> Dict[str, float]:
    scores = defaultdict(float)

    # 각 문서별 term frequency 저장
    doc_term_freqs = defaultdict(lambda: defaultdict(int))

    for term in query_tokens:
        # Posting list 조회
        postings = get_term_postings(cursor, term)

        if not postings:
            continue

        # IDF 계산
        df = len(postings)
        idf = calculate_idf(N, df)

        # 각 문서에 대해 BM25 점수 계산
        for doc_id, tf in postings.items():
            doc_term_freqs[doc_id][term] = tf

            # 문서 길이 조회
            doc_length = get_document_length(cursor, doc_id)

            numerator = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * (doc_length / avgdl))

            scores[doc_id] += idf * (numerator / denominator)

    return scores, doc_term_freqs

# 단일 쿼리 검색
def search_query(query_id: str, query_text: str, kiwi: Kiwi, cursor, N: int, avgdl: float) -> Dict:
    # 쿼리 토큰화
    query_tokens = tokenize(query_text, kiwi)

    if not query_tokens:
        return {
            'query_id': query_id,
            'query_text': query_text,
            'query_tokens': [],
            'num_results': 0,
            'results': []
        }

    # BIM 검색
    bim_scores = calculate_bim_score(query_tokens, cursor, N)

    # BM25 검색
    bm25_scores, doc_term_freqs = calculate_bm25_score(query_tokens, cursor, N, avgdl)

    # 모든 검색된 문서 ID 수집
    all_doc_ids = set(bim_scores.keys()) | set(bm25_scores.keys())

    # 결과 생성
    results = []
    for doc_id in all_doc_ids:
        doc_info = get_document_info(cursor, doc_id)
        if not doc_info:
            continue

        bim_score = bim_scores.get(doc_id, 0.0)
        bm25_score = bm25_scores.get(doc_id, 0.0)

        # 매칭된 용어 수 계산
        matched_terms = len([t for t in query_tokens if doc_id in get_term_postings(cursor, t)])

        # 용어 빈도 정보
        term_frequencies = doc_term_freqs.get(doc_id, {})

        results.append({
            'doc_id': doc_id,
            'doc_length': doc_info['length'],
            'doc_title': doc_info['title'],
            'bim_score': round(bim_score, 4),
            'bm25_score': round(bm25_score, 4),
            'score_difference': round(bm25_score - bim_score, 4),
            'matched_terms': matched_terms,
            'term_frequencies': term_frequencies
        })

    # BM25 score 기준 정렬
    results.sort(key=lambda x: x['bm25_score'], reverse=True)

    # TOP_K만 선택하고 순위 부여
    top_results = results[:TOP_K]
    for rank, result in enumerate(top_results, 1):
        result['rank'] = rank

    return {
        'query_id': query_id,
        'query_text': query_text,
        'query_tokens': query_tokens,
        'num_results': len(top_results),
        'results': top_results
    }

# 배치 검색
def batch_search(queries: List, kiwi: Kiwi, cursor, N: int, avgdl: float) -> List[Dict]:
    all_results = []

    for query in tqdm(queries, desc="검색 수행"):
        query_id = query['_id']
        query_text = query['text']

        result = search_query(query_id, query_text, kiwi, cursor, N, avgdl)
        all_results.append(result)

    return all_results

# 검색 결과 저장
def save_results(results: List[Dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(RESULTS_DIR, 'search_results.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n검색 결과 저장 완료: {output_path}")
    print(f"  총 {len(results):,}개 쿼리 처리\n")

# 샘플 결과 출력
def show_sample_results(results: List[Dict]):
    if not results:
        return

    print("\n[ 샘플 검색 결과 ]")
    sample = results[0]

    print(f"\nQuery ID: {sample['query_id']}")
    print(f"Query: {sample['query_text']}")
    print(f"Tokens: {sample['query_tokens']}")
    print(f"검색 결과: {sample['num_results']}개\n")

    print("상위 5개 문서:")
    for i, result in enumerate(sample['results'][:5], 1):
        print(f"\n  {i}. [{result['doc_id']}] {result['doc_title']}")
        print(f"     문서 길이: {result['doc_length']} 토큰")
        print(f"     BIM:  {result['bim_score']:.4f}")
        print(f"     BM25: {result['bm25_score']:.4f}")
        print(f"     차이: {result['score_difference']:.4f}")
        print(f"     매칭된 용어: {result['matched_terms']}개")

# main
def main():

    # 데이터 로드
    queries = load_queries()

    # Kiwi 초기화
    kiwi = Kiwi()

    # DB 연결
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 통계 정보 조회
    N, avgdl = get_statistics(cursor)
    print(f"통계 정보:")
    print(f"  N (전체 문서 수): {N:,}개")
    print(f"  avgdl (평균 문서 길이): {avgdl:.1f} 토큰\n")

    # 배치 검색 실행
    results = batch_search(queries, kiwi, cursor, N, avgdl)

    # 결과 저장
    save_results(results)

    # 샘플 결과 출력
    show_sample_results(results)

    # 연결 종료
    conn.close()
    print("\n검색 완료")

if __name__ == "__main__":
    main()