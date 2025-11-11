# visualize.py: BIM vs BM25 비교 분석 시각화 (로그 추가)

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR

ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOG_DIR = PROJECT_ROOT / 'logs'

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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

# 데이터 로드
def load_data():
    print("데이터 로드 시작\n")

    # 평가 결과
    with open(os.path.join(ANALYSIS_DIR, 'evaluation_results.json'), 'r', encoding='utf-8') as f:
        evaluation = json.load(f)
    print("  evaluation_results.json 로드 완료")

    # 선정된 쿼리
    with open(os.path.join(ANALYSIS_DIR, 'selected_queries.json'), 'r', encoding='utf-8') as f:
        selected_queries = json.load(f)
    print("  selected_queries.json 로드 완료")

    # 검색 결과
    with open(os.path.join(RESULTS_DIR, 'search_results.json'), 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    print("  search_results.json 로드 완료")

    print("\n데이터 로드 완료\n")
    return evaluation, selected_queries, search_results

# BIM vs BM25 성능 비교 그래프
def plot_performance_comparison(evaluation: dict):
    print("1. 성능 비교 그래프 생성 시작")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BIM vs BM25 성능 비교', fontsize=16, fontweight='bold')

    bim_metrics = evaluation['BIM']
    bm25_metrics = evaluation['BM25']

    k_values = [5, 10, 20, 50, 100]

    # Precision@K
    ax1 = axes[0, 0]
    bim_precision = [bim_metrics[f'P@{k}'] for k in k_values]
    bm25_precision = [bm25_metrics[f'P@{k}'] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    ax1.bar(x - width/2, bim_precision, width, label='BIM', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, bm25_precision, width, label='BM25', alpha=0.8, color='coral')

    ax1.set_xlabel('K', fontsize=11)
    ax1.set_ylabel('Precision', fontsize=11)
    ax1.set_title('Precision@K 비교', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Recall@K
    ax2 = axes[0, 1]
    bim_recall = [bim_metrics[f'R@{k}'] for k in k_values]
    bm25_recall = [bm25_metrics[f'R@{k}'] for k in k_values]

    ax2.bar(x - width/2, bim_recall, width, label='BIM', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, bm25_recall, width, label='BM25', alpha=0.8, color='coral')

    ax2.set_xlabel('K', fontsize=11)
    ax2.set_ylabel('Recall', fontsize=11)
    ax2.set_title('Recall@K 비교', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # F1@K
    ax3 = axes[1, 0]
    bim_f1 = [bim_metrics[f'F1@{k}'] for k in k_values]
    bm25_f1 = [bm25_metrics[f'F1@{k}'] for k in k_values]

    ax3.bar(x - width/2, bim_f1, width, label='BIM', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, bm25_f1, width, label='BM25', alpha=0.8, color='coral')

    ax3.set_xlabel('K', fontsize=11)
    ax3.set_ylabel('F1-Score', fontsize=11)
    ax3.set_title('F1@K 비교', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(k_values)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # MAP
    ax4 = axes[1, 1]
    models = ['BIM', 'BM25']
    map_values = [bim_metrics['MAP'], bm25_metrics['MAP']]
    colors = ['skyblue', 'coral']

    bars = ax4.bar(models, map_values, alpha=0.8, color=colors)
    ax4.set_ylabel('MAP', fontsize=11)
    ax4.set_title('Mean Average Precision 비교', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, '1_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   저장: {save_path}")
    plt.close()

# TF 효과 분석 그래프
def plot_tf_effect(selected_queries: dict, search_results: dict):
    print("2. TF 효과 분석 그래프 생성 시작")

    tf_queries = selected_queries['tf_effect']

    if not tf_queries:
        print("   TF 효과 쿼리 없음 - 스킵")
        return

    # 쿼리별 점수 차이 분석
    tf_values = []
    score_diffs = []
    query_labels = []

    for i, q_info in enumerate(tf_queries[:10], 1):
        query_id = q_info['query']['query_id']
        query_text = q_info['query']['query_text']

        # 해당 쿼리의 검색 결과 찾기
        query_result = next((qr for qr in search_results if qr['query_id'] == query_id), None)
        if not query_result:
            continue

        # 정답 문서들의 평균 점수 차이
        relevant_docs = set(q_info['relevant_docs'])
        relevant_results = [r for r in query_result['results'] if r['doc_id'] in relevant_docs]

        if relevant_results:
            avg_tf = np.mean([max(r['term_frequencies'].values()) if r['term_frequencies'] else 0
                              for r in relevant_results])
            avg_score_diff = np.mean([r['score_difference'] for r in relevant_results])

            tf_values.append(avg_tf)
            score_diffs.append(avg_score_diff)
            query_labels.append(f"Q{i}")

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TF(Term Frequency) 효과 분석', fontsize=16, fontweight='bold')

    # TF vs Score Difference (scatter)
    ax1.scatter(tf_values, score_diffs, s=100, alpha=0.6, color='coral')

    # 추세선
    if len(tf_values) > 1:
        z = np.polyfit(tf_values, score_diffs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tf_values), max(tf_values), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='추세선')

    ax1.set_xlabel('평균 Term Frequency', fontsize=11)
    ax1.set_ylabel('BM25 - BIM 점수 차이', fontsize=11)
    ax1.set_title('TF와 점수 차이 상관관계', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Query별 점수 차이
    ax2.bar(query_labels, score_diffs, alpha=0.8, color='coral')
    ax2.set_xlabel('쿼리', fontsize=11)
    ax2.set_ylabel('BM25 - BIM 평균 점수 차이', fontsize=11)
    ax2.set_title('쿼리별 점수 차이 (High TF 문서)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, '2_tf_effect_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   저장: {save_path}")
    plt.close()

# 문서 길이 정규화 효과 분석 그래프
def plot_length_effect(selected_queries: dict, search_results: dict):
    print("3. 길이 정규화 효과 분석 그래프 생성 시작")

    length_queries = selected_queries['length_effect']

    if not length_queries:
        print("   길이 효과 쿼리 없음 - 스킵")
        return

    # 쿼리별 짧은 문서 vs 긴 문서 점수 비교
    short_bim_scores = []
    short_bm25_scores = []
    long_bim_scores = []
    long_bm25_scores = []
    query_labels = []

    for i, q_info in enumerate(length_queries[:10], 1):
        query_id = q_info['query']['query_id']

        # 해당 쿼리의 검색 결과 찾기
        query_result = next((qr for qr in search_results if qr['query_id'] == query_id), None)
        if not query_result:
            continue

        relevant_docs = set(q_info['relevant_docs'])
        relevant_results = [r for r in query_result['results'] if r['doc_id'] in relevant_docs]

        # 짧은 문서와 긴 문서 분리
        short_docs = [r for r in relevant_results if r['doc_length'] < 500]
        long_docs = [r for r in relevant_results if r['doc_length'] > 2000]

        if short_docs and long_docs:
            short_bim_scores.append(np.mean([r['bim_score'] for r in short_docs]))
            short_bm25_scores.append(np.mean([r['bm25_score'] for r in short_docs]))
            long_bim_scores.append(np.mean([r['bim_score'] for r in long_docs]))
            long_bm25_scores.append(np.mean([r['bm25_score'] for r in long_docs]))
            query_labels.append(f"Q{i}")

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('문서 길이 정규화 효과 분석', fontsize=16, fontweight='bold')

    x = np.arange(len(query_labels))
    width = 0.35

    # 짧은 문서 점수 비교
    ax1.bar(x - width/2, short_bim_scores, width, label='BIM', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, short_bm25_scores, width, label='BM25', alpha=0.8, color='coral')

    ax1.set_xlabel('쿼리', fontsize=11)
    ax1.set_ylabel('평균 점수', fontsize=11)
    ax1.set_title('짧은 문서 (<500 토큰) 점수 비교', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 긴 문서 점수 비교
    ax2.bar(x - width/2, long_bim_scores, width, label='BIM', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, long_bm25_scores, width, label='BM25', alpha=0.8, color='coral')

    ax2.set_xlabel('쿼리', fontsize=11)
    ax2.set_ylabel('평균 점수', fontsize=11)
    ax2.set_title('긴 문서 (>2000 토큰) 점수 비교', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(query_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, '3_length_effect_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   저장: {save_path}")
    plt.close()

# 점수 분포 비교 그래프
def plot_score_distribution(search_results: list):
    print("4. 점수 분포 분석 그래프 생성 시작")

    all_bim_scores = []
    all_bm25_scores = []
    all_score_diffs = []

    for query_result in search_results:
        for result in query_result['results'][:20]:  # Top 20만
            all_bim_scores.append(result['bim_score'])
            all_bm25_scores.append(result['bm25_score'])
            all_score_diffs.append(result['score_difference'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('점수 분포 분석', fontsize=16, fontweight='bold')

    # BIM 점수 분포
    ax1 = axes[0, 0]
    ax1.hist(all_bim_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('BIM Score', fontsize=11)
    ax1.set_ylabel('빈도', fontsize=11)
    ax1.set_title('BIM 점수 분포', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(all_bim_scores), color='red', linestyle='--',
                linewidth=2, label=f'평균: {np.mean(all_bim_scores):.2f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # BM25 점수 분포
    ax2 = axes[0, 1]
    ax2.hist(all_bm25_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('BM25 Score', fontsize=11)
    ax2.set_ylabel('빈도', fontsize=11)
    ax2.set_title('BM25 점수 분포', fontsize=12, fontweight='bold')
    ax2.axvline(np.mean(all_bm25_scores), color='red', linestyle='--',
                linewidth=2, label=f'평균: {np.mean(all_bm25_scores):.2f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 점수 차이 분포
    ax3 = axes[1, 0]
    ax3.hist(all_score_diffs, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Score Difference (BM25 - BIM)', fontsize=11)
    ax3.set_ylabel('빈도', fontsize=11)
    ax3.set_title('점수 차이 분포', fontsize=12, fontweight='bold')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(np.mean(all_score_diffs), color='red', linestyle='--',
                linewidth=2, label=f'평균: {np.mean(all_score_diffs):.2f}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # BIM vs BM25 scatter plot
    ax4 = axes[1, 1]

    # 샘플링
    sample_size = min(5000, len(all_bim_scores))
    indices = np.random.choice(len(all_bim_scores), sample_size, replace=False)
    sampled_bim = [all_bim_scores[i] for i in indices]
    sampled_bm25 = [all_bm25_scores[i] for i in indices]

    ax4.scatter(sampled_bim, sampled_bm25, alpha=0.3, s=20, color='purple')

    # 대각선 (BIM = BM25)
    max_val = max(max(sampled_bim), max(sampled_bm25))
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='BIM = BM25')

    ax4.set_xlabel('BIM Score', fontsize=11)
    ax4.set_ylabel('BM25 Score', fontsize=11)
    ax4.set_title('BIM vs BM25 점수 상관관계', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, '4_score_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   저장: {save_path}")
    plt.close()

# 지표별 개선율 그래프
def plot_metric_improvements(evaluation: dict):
    print("5. 개선율 비교 그래프 생성 시작")

    bim_metrics = evaluation['BIM']
    bm25_metrics = evaluation['BM25']

    k_values = [5, 10, 20, 50, 100]

    # 개선율 계산
    precision_improvements = []
    recall_improvements = []
    f1_improvements = []

    for k in k_values:
        p_imp = ((bm25_metrics[f'P@{k}'] - bim_metrics[f'P@{k}']) / bim_metrics[f'P@{k}']) * 100 if bim_metrics[f'P@{k}'] > 0 else 0
        r_imp = ((bm25_metrics[f'R@{k}'] - bim_metrics[f'R@{k}']) / bim_metrics[f'R@{k}']) * 100 if bim_metrics[f'R@{k}'] > 0 else 0
        f1_imp = ((bm25_metrics[f'F1@{k}'] - bim_metrics[f'F1@{k}']) / bim_metrics[f'F1@{k}']) * 100 if bim_metrics[f'F1@{k}'] > 0 else 0

        precision_improvements.append(p_imp)
        recall_improvements.append(r_imp)
        f1_improvements.append(f1_imp)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(k_values))
    width = 0.25

    ax.bar(x - width, precision_improvements, width, label='Precision', alpha=0.8, color='skyblue')
    ax.bar(x, recall_improvements, width, label='Recall', alpha=0.8, color='coral')
    ax.bar(x + width, f1_improvements, width, label='F1-Score', alpha=0.8, color='lightgreen')

    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('개선율 (%)', fontsize=12)
    ax.set_title('BM25의 BIM 대비 성능 개선율', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, '5_improvement_rates.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   저장: {save_path}")
    plt.close()

# main
def main():
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'visualize_{timestamp}.log')
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("BIM vs BM25 비교 분석 시각화")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # 데이터 로드
        evaluation, selected_queries, search_results = load_data()

        # 그래프 생성
        print("그래프 생성 시작\n")

        # 1. 성능 비교
        plot_performance_comparison(evaluation)

        # 2. TF 효과
        plot_tf_effect(selected_queries, search_results)

        # 3. 길이 효과
        plot_length_effect(selected_queries, search_results)

        # 4. 점수 분포
        plot_score_distribution(search_results)

        # 5. 개선율
        plot_metric_improvements(evaluation)

        print("\n" + "=" * 60)
        print("생성된 그래프:")
        print("=" * 60)
        print("  1_performance_comparison.png: 전체 성능 비교")
        print("  2_tf_effect_analysis.png: TF 효과 분석")
        print("  3_length_effect_analysis.png: 길이 정규화 효과")
        print("  4_score_distribution.png: 점수 분포 분석")
        print("  5_improvement_rates.png: 개선율 비교")
        print("=" * 60 + "\n")

        print("시각화 완료")
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