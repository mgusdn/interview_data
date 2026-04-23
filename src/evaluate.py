# src/evaluate.py


import sys
sys.path.insert(0, 'src')
from retrieve import retrieve

test_queries = [
    ('PCA 차원 축소', 'machine-learning', 'PCA'),
    ('TCP UDP 차이', 'network', 'TCP'),
    ('gradient descent란', 'deep-learning', 'gradient'),
    ('파이썬 GIL이란', 'python', 'GIL'),
    ('정규분포란', 'statistics-math', '정규'),
    ('binary search 구현', 'algorithm', 'binary'),
    ('스택과 큐의 차이', 'data-structure', '스택'),
    ('프로세스와 스레드 차이', 'operating-system', '프로세스'),
    ('overfitting 방지 방법', 'machine-learning', 'overfitting'),
    ('dropout이란', 'deep-learning', 'dropout'),
    ('리스트와 튜플 차이', 'python', '리스트'),
    ('HTTP HTTPS 차이', 'network', 'HTTP'),
]

for query, category, keyword in test_queries:
    results = retrieve(query, k=5, category=category, threshold=0.3)
    top1 = results[0] if results else None
    top1_score = top1['score'] if top1 else 0
    top1_question = top1['question'][:40] if top1 else '결과 없음'
    hit = 'O' if top1 and keyword.lower() in top1['question'].lower() else 'X'
    print(hit, '|', query, '|', len(results), '|', top1_score, '|', top1_question)