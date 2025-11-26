# 터미널: pip install streamlit
# 실행: streamlit run app.py

import streamlit as st
import sqlite3
import math
import pandas as pd
from kiwipiepy import Kiwi
from collections import defaultdict
from pathlib import Path

# 1. 설정 및 초기화
st.set_page_config(layout="wide", page_title="한국어 검색 엔진")

# 경로 설정
DB_PATH = Path('database/inverted_index_backup.db')

# Kiwi 초기화
@st.cache_resource
def get_kiwi():
    return Kiwi(num_workers=-1)

kiwi = get_kiwi()

# DB 연결
def get_db_connection():
    return sqlite3.connect(str(DB_PATH))


# 2. 검색 엔진
def tokenize(text):
    if not text: return []
    clean_text = text.replace('\x00', '')
    try:
        tokens = kiwi.tokenize(clean_text)
        useful_tags = ['NNG', 'NNP', 'VV', 'VA', 'MAG']
        return [t.form for t in tokens if t.tag in useful_tags and len(t.form) > 1]
    except:
        return []

def search(query):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT value FROM statistics WHERE key='N'")
        N = cursor.fetchone()[0]
        cursor.execute("SELECT value FROM statistics WHERE key='avgdl'")
        avgdl = cursor.fetchone()[0]
    except:
        st.error("DB가 비어있거나 통계 정보가 없습니다. 01번 코드를 먼저 실행하세요.")
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    K1 = 1.2
    B = 0.75

    bim_scores = defaultdict(float)
    bm25_scores = defaultdict(float)
    doc_details = {}

    for term in q_tokens:
        cursor.execute("SELECT doc_id, tf FROM inverted_index WHERE term = ?", (term,))
        postings = cursor.fetchall()
        if not postings: continue

        df = len(postings)
        idf = math.log((N - df + 0.5) / (df + 0.5))
        if idf < 0: idf = 0

        for doc_id, tf in postings:
            if doc_id not in doc_details:
                cursor.execute("SELECT title, length FROM documents WHERE doc_id = ?", (doc_id,))
                res = cursor.fetchone()
                if res:
                    doc_details[doc_id] = {'title': res[0], 'length': res[1]}

            if doc_id not in doc_details: continue

            doc_len = doc_details[doc_id]['length']

            # BIM Score
            bim_scores[doc_id] += idf

            # BM25 Score
            numerator = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * (doc_len / avgdl))
            bm25_scores[doc_id] += idf * (numerator / denominator)

    conn.close()

    results = []
    all_doc_ids = set(bim_scores.keys()) | set(bm25_scores.keys())

    for doc_id in all_doc_ids:
        if doc_id not in doc_details: continue
        results.append({
            'Doc ID': doc_id,
            'Title': doc_details[doc_id]['title'],
            'Length': doc_details[doc_id]['length'],
            'BIM Score': round(bim_scores.get(doc_id, 0), 4),
            'BM25 Score': round(bm25_scores.get(doc_id, 0), 4),
            'Difference': round(bm25_scores.get(doc_id, 0) - bim_scores.get(doc_id, 0), 4)
        })

    # BM25 기준 내림차순 정렬
    results.sort(key=lambda x: x['BM25 Score'], reverse=True)
    return results[:50] # 상위 50개만 리턴


# 3. UI 레이아웃
st.title("한국어 검색 엔진 (BIM vs BM25)")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("검색어를 입력하세요", placeholder="예: 이화여자대학교는 스크랜튼 여사가 1886년에 설립했다.")
with col2:
    st.write("")
    st.write("")
    search_btn = st.button("검색", type="primary")

if search_btn or query:
    if not query.strip():
        st.warning("검색어를 입력하세요.")
    else:
        with st.spinner('문서를 검색 중입니다.'):
            results = search(query)

        if not results:
            st.info("검색 결과가 없습니다.")
        else:
            st.success(f"총 {len(results)}개의 문서를 찾았습니다.")

            df = pd.DataFrame(results)

            tab1, tab2 = st.tabs(["요약 테이블", "상세 결과"])

            with tab1:
                st.dataframe(
                    df[['Title', 'BM25 Score', 'BIM Score', 'Difference', 'Length']],
                    use_container_width=True
                )

            with tab2:
                for i, row in df.iterrows():
                    with st.expander(f"[{i+1}위] {row['Title']} (BM25: {row['BM25 Score']})"):
                        st.write(f"**Document ID:** {row['Doc ID']}")
                        st.write(f"**Length:** {row['Length']} tokens")
                        st.metric(label="Score Difference (BM25 - BIM)", value=row['Difference'])