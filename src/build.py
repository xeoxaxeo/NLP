# build.py: Inverted Index 구축 및 SQLite 저장

import os
import sqlite3
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List
from tqdm import tqdm
from kiwipiepy import Kiwi
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / 'data'
INDEX_DIR = PROJECT_ROOT / 'index'
DB_PATH = INDEX_DIR / 'inverted_index.db'
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


# 데이터 로드
def load_data():
    with open(os.path.join(DATA_DIR, 'corpus.pkl'), 'rb') as f:
        corpus = pickle.load(f)
    print(f"corpus.pkl 로드 ({len(corpus):,}개)")

    print("데이터 로드 완료\n")
    return corpus


# SQLite DB 초기화
def initialize_database():
    # 기존 DB 삭제
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # DB 연결
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.execute("""
    CREATE TABLE documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT,
        length INTEGER,
        tokens TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE inverted_index (
        term TEXT,
        doc_id TEXT,
        tf INTEGER,
        PRIMARY KEY (term, doc_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE statistics (
        key TEXT PRIMARY KEY,
        value REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE term_stats (
        term TEXT PRIMARY KEY,
        df INTEGER
    )
    """)

    conn.commit()
    print("테이블 생성 완료\n")

    return conn, cursor


# 문서 전처리 및 토큰화
def tokenize_documents(corpus: List, kiwi: Kiwi):

    doc_tokens = {}
    doc_lengths = {}

    for doc in tqdm(corpus, desc="Tokening"):
        doc_id = doc['_id']
        title = doc['title']
        text = doc['text']

        # 제목 + 본문 토큰화
        full_text = f"{title} {text}"
        tokens = tokenize(full_text, kiwi)

        doc_tokens[doc_id] = tokens
        doc_lengths[doc_id] = len(tokens)

    print(f"\n{len(doc_tokens):,}개 문서 토큰화 완료")
    print(f"  평균 문서 길이: {sum(doc_lengths.values()) / len(doc_lengths):.1f} 토큰\n")

    return doc_tokens, doc_lengths


# Inverted Index 구축
def build_inverted_index(doc_tokens: Dict):
    # Inverted Index: {term: {doc_id: tf}}
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, tokens in tqdm(doc_tokens.items(), desc="Indexing"):
        # 각 토큰의 빈도 계산
        token_counts = Counter(tokens)

        for term, tf in token_counts.items():
            inverted_index[term][doc_id] = tf

    print(f"\nInverted Index 구축 완료")
    print(f"  전체 용어 수: {len(inverted_index):,}개\n")

    return inverted_index


# SQLite에 저장
def save_to_database(conn, cursor, corpus, doc_tokens, doc_lengths, inverted_index):

    # Documents 테이블
    doc_data = []
    for doc_id, tokens in doc_tokens.items():
        title = next((d['title'] for d in corpus if d['_id'] == doc_id), "")
        doc_data.append((
            doc_id,
            title,
            len(tokens),
            json.dumps(tokens, ensure_ascii=False)
        ))

    cursor.executemany(
        "INSERT INTO documents VALUES (?, ?, ?, ?)",
        doc_data
    )
    conn.commit()
    print(f"    {len(doc_data):,}개 문서 저장")

    # Inverted Index 테이블
    index_data = []
    for term, postings in tqdm(inverted_index.items(), desc="Index 저장"):
        for doc_id, tf in postings.items():
            index_data.append((term, doc_id, tf))

    cursor.executemany(
        "INSERT INTO inverted_index VALUES (?, ?, ?)",
        index_data
    )
    conn.commit()
    print(f"    {len(index_data):,}개 posting 저장")

    # Term Statistics 테이블
    term_stats_data = []
    for term, postings in inverted_index.items():
        df = len(postings)
        term_stats_data.append((term, df))

    cursor.executemany(
        "INSERT INTO term_stats VALUES (?, ?)",
        term_stats_data
    )
    conn.commit()
    print(f"    {len(term_stats_data):,}개 term stats 저장")

    # Statistics 테이블
    N = len(doc_tokens)
    avgdl = sum(doc_lengths.values()) / N

    stats_data = [
        ('N', N),
        ('avgdl', avgdl),
        ('total_terms', len(inverted_index))
    ]

    cursor.executemany(
        "INSERT INTO statistics VALUES (?, ?)",
        stats_data
    )
    conn.commit()
    print(f"    통계 정보 저장")

    print("\n모든 데이터 저장 완료\n")


# DB 인덱스 생성
def create_indexes(conn, cursor):
    cursor.execute("CREATE INDEX idx_term ON inverted_index(term)")
    cursor.execute("CREATE INDEX idx_doc_id ON inverted_index(doc_id)")
    conn.commit()
    print("인덱스 생성 완료\n")


# 통계 정보 출력
def show_statistics(cursor):

    cursor.execute("SELECT value FROM statistics WHERE key='N'")
    N = cursor.fetchone()[0]

    cursor.execute("SELECT value FROM statistics WHERE key='avgdl'")
    avgdl = cursor.fetchone()[0]

    cursor.execute("SELECT value FROM statistics WHERE key='total_terms'")
    total_terms = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM inverted_index")
    total_postings = cursor.fetchone()[0]

    print(f"  전체 문서 수 (N): {int(N):,}개")
    print(f"  평균 문서 길이 (avgdl): {avgdl:.1f} 토큰")
    print(f"  전체 용어 수: {int(total_terms):,}개")
    print(f"  전체 posting 수: {total_postings:,}개")

    # 상위 빈도 용어 확인
    print("\n  [ 상위 10개 고빈도 용어 ]")
    cursor.execute("""
        SELECT term, df 
        FROM term_stats 
        ORDER BY df DESC 
        LIMIT 10
    """)
    for i, (term, df) in enumerate(cursor.fetchall(), 1):
        print(f"    {i}. '{term}': {df}개 문서")

    print()


# main
def main():
    # 데이터 로드
    corpus = load_data()

    # Kiwi 초기화
    kiwi = Kiwi()

    # DB 초기화
    conn, cursor = initialize_database()

    # 문서 토큰화
    doc_tokens, doc_lengths = tokenize_documents(corpus, kiwi)

    # Inverted Index 구축
    inverted_index = build_inverted_index(doc_tokens)

    # SQLite에 저장
    save_to_database(conn, cursor, corpus, doc_tokens, doc_lengths, inverted_index)

    # DB 인덱스 생성
    create_indexes(conn, cursor)

    # 통계 출력
    show_statistics(cursor)

    # 연결 종료
    conn.close()

if __name__ == "__main__":
    main()