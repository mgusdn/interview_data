# src/test_rewrite.py
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, 'src')
from retrieve import retrieve

load_dotenv()
client = OpenAI()

CATEGORIES = [
    "statistics-math", "machine-learning", "deep-learning",
    "python", "network", "operating-system", "data-structure", "algorithm",
]

def analyze_query(user_query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""다음 질문을 분석해주세요.

질문: {user_query}

아래 JSON 형식으로만 답변하세요:
{{
    "category": "아래 카테고리 중 하나",
    "rewritten_query": "면접 질문 스타일로 재작성한 쿼리"
}}

카테고리 설명:
- statistics-math: 통계, 확률, 수학, 분포, 검정
- machine-learning: 머신러닝 알고리즘, 모델, overfitting, 정규화, 앙상블
- deep-learning: 신경망, 딥러닝, gradient descent, activation function, dropout
- python: 파이썬 언어, GIL, 자료형, 라이브러리, 문법
- network: TCP, UDP, HTTP, HTTPS, 네트워크 프로토콜
- operating-system: OS, 프로세스, 스레드, 메모리, 데드락
- data-structure: 자료구조, 스택, 큐, 트리, 해시
- algorithm: 정렬, 탐색, DP, 그래프 알고리즘"""
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

test_queries = [
    "overfitting 방지 방법",
    "TCP UDP 차이",
    "GIL이란",
    "gradient descent란",
    "HTTP HTTPS 차이",
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"원본 쿼리: {query}")
    analysis = analyze_query(query)
    print(f"카테고리: {analysis['category']}")
    print(f"재작성: {analysis['rewritten_query']}")
    
    results = retrieve(analysis['rewritten_query'], k=3, category=analysis['category'], threshold=0.3)
    for r in results:
        print(f"  [{r['rank']}] {r['question']} | score: {r['score']}")
    if not results:
        print("  결과 없음")