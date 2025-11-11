# build_multiprocsssing.py: Inverted Index 구축 (멀티프로세싱 ver.)

import os
import sys
import sqlite3
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
from kiwipiepy import Kiwi
from pathlib import Path
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / 'data'
INDEX_DIR = PROJECT_ROOT / 'index'
LOG_DIR = PROJECT_ROOT / 'logs'
DB_PATH = INDEX_DIR / 'inverted_index.db'

# 사용할 프로세스 수 (CPU 코어 수 - 1)
NUM_PROCESSES = max(1, cpu_count() - 1)

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
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("기존 데이터베이스 삭제\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

# 단일 문서 토큰화
def tokenize_single_doc(doc):
    kiwi = Kiwi()
    doc_id = doc['_id']
    title = doc['title']
    text = doc['text']

    full_text = f"{title} {text}"
    tokens = tokenize(full_text, kiwi)

    return doc_id, tokens

# 문서 전처리 및 토큰화
def tokenize_documents(corpus: List):
    print(f"총 {len(corpus):,}개 문서 토큰화 시작 ({NUM_PROCESSES}개 프로세스 사용)")

    doc_tokens = {}
    doc_lengths = {}

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(tokenize_single_doc, corpus),
            total=len(corpus),
            desc="문서 토큰화"
        ))

    # 결과 정리
    for doc_id, tokens in results:
        doc_tokens[doc_id] = tokens
        doc_lengths[doc_id] = len(tokens)

    avg_length = sum(doc_lengths.values()) / len(doc_lengths)
    print(f"\n토큰화 완료:")
    print(f"  - 문서 수: {len(doc_tokens):,}개")
    print(f"  - 평균 문서 길이: {avg_length:.1f} 토큰")
    print(f"  - 최소 길이: {min(doc_lengths.values())} 토큰")
    print(f"  - 최대 길이: {max(doc_lengths.values())} 토큰\n")

    return doc_tokens, doc_lengths

# Inverted Index 구축
def build_inverted_index(doc_tokens: Dict):
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, tokens in tqdm(doc_tokens.items(), desc="인덱스 구축"):
        token_counts = Counter(tokens)

        for term, tf in token_counts.items():
            inverted_index[term][doc_id] = tf

    print(f"\nInverted Index 구축 완료:")
    print(f"  - 전체 용어 수: {len(inverted_index):,}개")

    # 통계 출력
    posting_counts = [len(postings) for postings in inverted_index.values()]
    print(f"  - 평균 posting list 길이: {sum(posting_counts)/len(posting_counts):.1f}")
    print(f"  - 최대 posting list 길이: {max(posting_counts)}\n")

    return inverted_index

# SQLite에 저장
def save_to_database(conn, cursor, corpus, doc_tokens, doc_lengths, inverted_index):
    print("데이터베이스 저장 시작\n")

    # Documents 테이블
    print("1. Documents 테이블 저장 시작")
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
    print(f"   완료: {len(doc_data):,}개 문서\n")

    # Inverted Index 테이블
    print("2. Inverted Index 테이블 저장 시작")
    index_data = []
    for term, postings in tqdm(inverted_index.items(), desc="   저장 진행"):
        for doc_id, tf in postings.items():
            index_data.append((term, doc_id, tf))

    cursor.executemany(
        "INSERT INTO inverted_index VALUES (?, ?, ?)",
        index_data
    )
    conn.commit()
    print(f"   완료: {len(index_data):,}개 posting\n")

    # Term Statistics 테이블
    print("3. Term Statistics 저장 시작")
    term_stats_data = []
    for term, postings in inverted_index.items():
        df = len(postings)
        term_stats_data.append((term, df))

    cursor.executemany(
        "INSERT INTO term_stats VALUES (?, ?)",
        term_stats_data
    )
    conn.commit()
    print(f"   완료: {len(term_stats_data):,}개 term\n")

    # Statistics 테이블
    print("4. 통계 정보 저장 시작")
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
    print(f"   완료\n")

    print("모든 데이터 저장 완료\n")

# DB 인덱스 생성
def create_indexes(conn, cursor):
    print("데이터베이스 인덱스 생성 시작")
    cursor.execute("CREATE INDEX idx_term ON inverted_index(term)")
    cursor.execute("CREATE INDEX idx_doc_id ON inverted_index(doc_id)")
    conn.commit()
    print("인덱스 생성 완료\n")

# 통계 정보 출력
def show_statistics(cursor):
    print("=" * 60)
    print("최종 통계")
    print("=" * 60 + "\n")

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

    print("\n  [ 상위 10개 고빈도 용어 ]")
    cursor.execute("""
        SELECT term, df 
        FROM term_stats 
        ORDER BY df DESC 
        LIMIT 10
    """)
    for i, (term, df) in enumerate(cursor.fetchall(), 1):
        print(f"    {i}. '{term}': {df:,}개 문서")

    print()

# main
def main():
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'build_{timestamp}.log')
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("Inverted Index 구축")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"사용 프로세스: {NUM_PROCESSES}개")
        print("=" * 60 + "\n")

        # 데이터 로드
        corpus = load_data()

        # DB 초기화
        conn, cursor = initialize_database()

        # 문서 토큰화
        doc_tokens, doc_lengths = tokenize_documents(corpus)

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

        print("=" * 60)
        print("인덱스 구축 완료")
        print(f"데이터베이스: {DB_PATH}")
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