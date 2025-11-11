# search.py: BIM/BM25 검색 엔진 구현 (로그 추가)

import os
import sys
import sqlite3
import json
import pickle
import math
from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from kiwipiepy import Kiwi
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / 'data'
INDEX_DIR = PROJECT_ROOT / 'index'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOG_DIR = PROJECT_ROOT / 'logs'
DB_PATH = INDEX_DIR / 'inverted_index.db'

# BM25 파라미터
K1 = 1.2
B = 0.75

# 결과 반환 개수
TOP_K = 100

# 로그 클래스
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
        postings = get_term_postings(cursor, term)

        if not postings:
            continue

        df = len(postings)
        idf = calculate_idf(N, df)

        for doc_id in postings.keys():
            scores[doc_id] += idf

    return scores

# BM25 score 계산
def calculate_bm25_score(query_tokens: List[str], cursor, N: int, avgdl: float) -> tuple:
    scores = defaultdict(float)
    doc_term_freqs = defaultdict(lambda: defaultdict(int))

    for term in query_tokens:
        postings = get_term_postings(cursor, term)

        if not postings:
            continue

        df = len(postings)
        idf = calculate_idf(N, df)

        for doc_id, tf in postings.items():
            doc_term_freqs[doc_id][term] = tf

            doc_length = get_document_length(cursor, doc_id)

            numerator = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * (doc_length / avgdl))

            scores[doc_id] += idf * (numerator / denominator)

    return scores, doc_term_freqs

# 단일 쿼리 검색
def search_query(query_id: str, query_text: str, kiwi: Kiwi, cursor, N: int, avgdl: float) -> Dict:
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

        matched_terms = len([t for t in query_tokens if doc_id in get_term_postings(cursor, t)])
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

    print(f"총 {len(queries):,}개 쿼리 검색 시작\n")
    for query in tqdm(queries, desc="검색 수행"):
        query_id = query['_id']
        query_text = query['text']

        result = search_query(query_id, query_text, kiwi, cursor, N, avgdl)
        all_results.append(result)

    print(f"\n검색 완료: {len(all_results):,}개 쿼리 처리\n")
    return all_results

# 검색 결과 저장
def save_results(results: List[Dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(RESULTS_DIR, 'search_results.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"검색 결과 저장 완료: {output_path}")
    print(f"  총 {len(results):,}개 쿼리 처리\n")

# 샘플 결과 출력
def show_sample_results(results: List[Dict]):
    if not results:
        return

    print("=" * 60)
    print("샘플 검색 결과")
    print("=" * 60 + "\n")

    sample = results[0]

    print(f"Query ID: {sample['query_id']}")
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

    print()

# 검색 통계 출력
def print_search_statistics(results: List[Dict]):
    print("=" * 60)
    print("검색 통계")
    print("=" * 60 + "\n")

    total_queries = len(results)
    total_results = sum(r['num_results'] for r in results)
    avg_results = total_results / total_queries if total_queries > 0 else 0

    all_score_diffs = []
    for r in results:
        for doc in r['results']:
            all_score_diffs.append(doc['score_difference'])

    print(f"총 쿼리 수: {total_queries:,}개")
    print(f"총 검색 결과: {total_results:,}개")
    print(f"쿼리당 평균 결과: {avg_results:.1f}개")

    if all_score_diffs:
        import numpy as np
        print(f"\n점수 차이 (BM25 - BIM) 통계:")
        print(f"  평균: {np.mean(all_score_diffs):.4f}")
        print(f"  중앙값: {np.median(all_score_diffs):.4f}")
        print(f"  최소: {np.min(all_score_diffs):.4f}")
        print(f"  최대: {np.max(all_score_diffs):.4f}")

    print()

# main
def main():
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'search_{timestamp}.log')
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("BIM/BM25 검색 수행")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # 데이터 로드
        queries = load_queries()

        # Kiwi 초기화
        print("Kiwi 형태소 분석기 초기화 시작")
        kiwi = Kiwi()
        print("초기화 완료\n")

        # DB 연결
        print("데이터베이스 연결 시작")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print("연결 완료\n")

        # 통계 정보 조회
        N, avgdl = get_statistics(cursor)
        print(f"통계 정보:")
        print(f"  N (전체 문서 수): {N:,}개")
        print(f"  avgdl (평균 문서 길이): {avgdl:.1f} 토큰")
        print(f"  BM25 파라미터: K1={K1}, B={B}")
        print(f"  반환 결과 수: TOP {TOP_K}\n")

        # 배치 검색 실행
        results = batch_search(queries, kiwi, cursor, N, avgdl)

        # 결과 저장
        save_results(results)

        # 통계 출력
        print_search_statistics(results)

        # 샘플 결과 출력
        show_sample_results(results)

        # 연결 종료
        conn.close()

        print("=" * 60)
        print("검색 완료")
        print(f"로그 파일: {log_file}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n로그 저장: {log_file}")

if __name__ == "__main__":
    main()