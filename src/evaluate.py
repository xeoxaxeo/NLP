# evaluate.py: 검색 결과 평가 및 쿼리 선정

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

# 평가 K 값들
K_VALUES = [5, 10, 20, 50, 100]

# 데이터 로드
def load_data():
    # 검색 결과
    with open(os.path.join(RESULTS_DIR, 'search_results.json'), 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    print(f"검색 결과 로드: {len(search_results):,}개 쿼리\n")

    # Qrels (정답 데이터)
    with open(os.path.join(DATA_DIR, 'qrels.pkl'), 'rb') as f:
        qrels = pickle.load(f)
    print(f"Qrels 로드: {len(qrels):,}개\n")

    return search_results, qrels

# Qrels 딕셔너리 생성
def build_qrels_dict(qrels: List) -> Dict[str, Set[str]]:
    qrels_dict = defaultdict(set)

    for qrel in qrels:
        query_id = qrel['query-id']
        doc_id = qrel['corpus-id']
        qrels_dict[query_id].add(doc_id)

    return qrels_dict

# Precision@K 계산
def calculate_precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])

    return relevant_retrieved / k

# Recall@K 계산
def calculate_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0

    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])

    return relevant_retrieved / len(relevant)

# F1@K 계산
def calculate_f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

# Average Precision 계산
def calculate_average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    if len(relevant) == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i

    return precision_sum / len(relevant) if len(relevant) > 0 else 0.0

# 단일 쿼리 평가
def evaluate_single_query(query_result: Dict, relevant_docs: Set[str], model: str = 'bm25') -> Dict:
    # 검색 결과에서 doc_id 추출 (점수 기준 정렬)
    if model == 'bim':
        sorted_results = sorted(query_result['results'], key=lambda x: x['bim_score'], reverse=True)
    else:  # bm25
        sorted_results = sorted(query_result['results'], key=lambda x: x['bm25_score'], reverse=True)

    retrieved = [r['doc_id'] for r in sorted_results]

    metrics = {}
    for k in K_VALUES:
        precision = calculate_precision_at_k(retrieved, relevant_docs, k)
        recall = calculate_recall_at_k(retrieved, relevant_docs, k)
        f1 = calculate_f1_at_k(precision, recall)

        metrics[f'P@{k}'] = precision
        metrics[f'R@{k}'] = recall
        metrics[f'F1@{k}'] = f1

    # Average Precision
    metrics['AP'] = calculate_average_precision(retrieved, relevant_docs)

    return metrics

# 모든 쿼리 평가
def evaluate_all_queries(search_results: List[Dict], qrels_dict: Dict, model: str = 'bm25') -> Dict:
    all_metrics = defaultdict(list)

    for query_result in search_results:
        query_id = query_result['query_id']
        relevant_docs = qrels_dict.get(query_id, set())

        if len(relevant_docs) == 0:
            continue

        metrics = evaluate_single_query(query_result, relevant_docs, model)

        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)

    # 평균 계산
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        avg_metrics[metric_name] = np.mean(values)

    # MAP 계산
    avg_metrics['MAP'] = avg_metrics['AP']

    return avg_metrics

# BIM vs BM25
def compare_models(search_results: List[Dict], qrels_dict: Dict) -> Dict:

    bim_metrics = evaluate_all_queries(search_results, qrels_dict, model='bim')
    bm25_metrics = evaluate_all_queries(search_results, qrels_dict, model='bm25')

    comparison = {
        'BIM': bim_metrics,
        'BM25': bm25_metrics
    }

    return comparison

# 테스트 쿼리 선정
def select_test_queries(search_results: List[Dict], qrels_dict: Dict) -> Dict[str, List[Dict]]:

    # 문서 길이 평균 계산
    all_doc_lengths = []
    for query_result in search_results:
        for result in query_result['results']:
            all_doc_lengths.append(result['doc_length'])

    avgdl = np.mean(all_doc_lengths)
    print(f"평균 문서 길이: {avgdl:.1f} 토큰\n")

    tf_effect_queries = []
    length_effect_queries = []
    baseline_queries = []

    for query_result in search_results:
        query_id = query_result['query_id']
        relevant_docs = qrels_dict.get(query_id, set())

        if len(relevant_docs) == 0:
            continue

        # 정답 문서들 중에서 분석
        relevant_results = [r for r in query_result['results'] if r['doc_id'] in relevant_docs]

        if not relevant_results:
            continue

        # TF 효과 쿼리 (높은 TF)
        high_tf_docs = [r for r in relevant_results if any(tf >= 5 for tf in r['term_frequencies'].values())]
        if high_tf_docs and len(tf_effect_queries) < 10:
            max_tf = max(max(r['term_frequencies'].values()) if r['term_frequencies'] else 0 for r in high_tf_docs)
            tf_effect_queries.append({
                'query': query_result,
                'relevant_docs': list(relevant_docs),
                'max_tf': max_tf,
                'reason': 'high_tf'
            })

        # 길이 효과 쿼리 (짧은 문서 + 긴 문서)
        short_docs = [r for r in relevant_results if r['doc_length'] < 500]
        long_docs = [r for r in relevant_results if r['doc_length'] > 2000]

        if short_docs and long_docs and len(length_effect_queries) < 10:
            # TF 낮은 것만 선택
            low_tf_short = [r for r in short_docs if all(tf <= 2 for tf in r['term_frequencies'].values())]
            low_tf_long = [r for r in long_docs if all(tf <= 2 for tf in r['term_frequencies'].values())]

            if low_tf_short and low_tf_long:
                length_effect_queries.append({
                    'query': query_result,
                    'relevant_docs': list(relevant_docs),
                    'short_doc_length': min(r['doc_length'] for r in low_tf_short),
                    'long_doc_length': max(r['doc_length'] for r in low_tf_long),
                    'reason': 'length_effect'
                })

        # Baseline 쿼리 (대조군)
        normal_docs = [r for r in relevant_results
                       if avgdl - 300 <= r['doc_length'] <= avgdl + 300
                       and all(1 <= tf <= 2 for tf in r['term_frequencies'].values())]

        if normal_docs and len(baseline_queries) < 10:
            baseline_queries.append({
                'query': query_result,
                'relevant_docs': list(relevant_docs),
                'avg_doc_length': np.mean([r['doc_length'] for r in normal_docs]),
                'reason': 'baseline'
            })

    selected_queries = {
        'tf_effect': tf_effect_queries[:10],
        'length_effect': length_effect_queries[:10],
        'baseline': baseline_queries[:10]
    }

    print(f"선정된 쿼리:")
    print(f"  TF 효과: {len(selected_queries['tf_effect'])}개")
    print(f"  길이 효과: {len(selected_queries['length_effect'])}개")
    print(f"  Baseline: {len(selected_queries['baseline'])}개\n")

    return selected_queries

# 평가 결과 저장
def save_evaluation_results(comparison: Dict, selected_queries: Dict):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    eval_path = os.path.join(ANALYSIS_DIR, 'evaluation_results.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"평가 결과 저장: {eval_path}")

    queries_path = os.path.join(ANALYSIS_DIR, 'selected_queries.json')
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(selected_queries, f, ensure_ascii=False, indent=2)
    print(f"선정 쿼리 저장: {queries_path}\n")

# 비교 결과
def print_comparison_results(comparison: Dict):

    bim_metrics = comparison['BIM']
    bm25_metrics = comparison['BM25']

    print(f"{'Metric':<15} {'BIM':<12} {'BM25':<12} {'Diff':<12}")
    print("-" * 60)

    # Precision@K
    for k in K_VALUES:
        metric = f'P@{k}'
        bim_val = bim_metrics[metric]
        bm25_val = bm25_metrics[metric]
        diff = bm25_val - bim_val
        print(f"{metric:<15} {bim_val:<12.4f} {bm25_val:<12.4f} {diff:+.4f}")

    print()

    # Recall@K
    for k in K_VALUES:
        metric = f'R@{k}'
        bim_val = bim_metrics[metric]
        bm25_val = bm25_metrics[metric]
        diff = bm25_val - bim_val
        print(f"{metric:<15} {bim_val:<12.4f} {bm25_val:<12.4f} {diff:+.4f}")

    print()

    # F1@K
    for k in K_VALUES:
        metric = f'F1@{k}'
        bim_val = bim_metrics[metric]
        bm25_val = bm25_metrics[metric]
        diff = bm25_val - bim_val
        print(f"{metric:<15} {bim_val:<12.4f} {bm25_val:<12.4f} {diff:+.4f}")

    print()

    # MAP
    bim_map = bim_metrics['MAP']
    bm25_map = bm25_metrics['MAP']
    diff = bm25_map - bim_map
    print(f"{'MAP':<15} {bim_map:<12.4f} {bm25_map:<12.4f} {diff:+.4f}")
    print()

# 선정된 테스트 쿼리 샘플
def print_selected_queries_summary(selected_queries: Dict):

    if selected_queries['tf_effect']:
        print("[ TF 효과 쿼리 ]")
        for i, q in enumerate(selected_queries['tf_effect'][:3], 1):
            print(f"\n{i}. Query: {q['query']['query_text']}")
            print(f"   Max TF: {q['max_tf']}")
            print(f"   정답 문서: {len(q['relevant_docs'])}개")

    print()

    if selected_queries['length_effect']:
        print("[ 길이 효과 쿼리 ]")
        for i, q in enumerate(selected_queries['length_effect'][:3], 1):
            print(f"\n{i}. Query: {q['query']['query_text']}")
            print(f"   짧은 문서: {q['short_doc_length']} 토큰")
            print(f"   긴 문서: {q['long_doc_length']} 토큰")
            print(f"   정답 문서: {len(q['relevant_docs'])}개")

    print()

    if selected_queries['baseline']:
        print("[ Baseline 쿼리 ]")
        for i, q in enumerate(selected_queries['baseline'][:3], 1):
            print(f"\n{i}. Query: {q['query']['query_text']}")
            print(f"   평균 문서 길이: {q['avg_doc_length']:.1f} 토큰")
            print(f"   정답 문서: {len(q['relevant_docs'])}개")

    print()

# main
def main():
    print("=" * 60)
    print("검색 결과 평가 및 쿼리 선정")
    print("=" * 60 + "\n")

    # 데이터 로드
    search_results, qrels = load_data()

    # Qrels 딕셔너리 생성
    qrels_dict = build_qrels_dict(qrels)
    print(f"Qrels 딕셔너리 생성: {len(qrels_dict):,}개 쿼리\n")

    # 모델 비교 평가
    comparison = compare_models(search_results, qrels_dict)

    # 테스트 쿼리 선정
    selected_queries = select_test_queries(search_results, qrels_dict)

    # 결과 저장
    save_evaluation_results(comparison, selected_queries)

    # 결과 출력
    print_comparison_results(comparison)
    print_selected_queries_summary(selected_queries)

    print("\n평가 완료")


if __name__ == "__main__":
    main()