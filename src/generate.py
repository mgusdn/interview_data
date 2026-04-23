import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, 'src')
from retrieve import retrieve

load_dotenv()
client = OpenAI()

CATEGORIES = [
    "statistics-math",
    "machine-learning",
    "deep-learning",
    "python",
    "network",
    "operating-system",
    "data-structure",
    "algorithm",
]


def analyze_query(user_query: str) -> dict:
    """LLM이 카테고리 판단 + 쿼리 재작성"""
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
- algorithm: 정렬, 탐색, DP, 그래프 알고리즘

예시:
질문: "overfitting 방지 방법"
{{
    "category": "machine-learning",
    "rewritten_query": "오버피팅일 경우 어떻게 대처해야 할까요?"
}}"""
        }],
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)


def generate_answer(user_query: str) -> str:
    # 1. 카테고리 판단 + 쿼리 재작성
    print("쿼리 분석 중...")
    analysis = analyze_query(user_query)
    category = analysis["category"]
    rewritten_query = analysis["rewritten_query"]
    print(f"카테고리: {category}")
    print(f"재작성된 쿼리: {rewritten_query}")

    # 2. 검색
    results = retrieve(rewritten_query, k=5, category=category, threshold=0.5)

    if not results:
        # threshold 낮춰서 재시도
        results = retrieve(rewritten_query, k=5, category=category, threshold=0.3)

    if not results:
        return "관련 Q&A를 찾지 못했습니다."

    print(f"검색 결과: {len(results)}개")

    # 3. 컨텍스트 구성
    context = ""
    for r in results:
        context += f"Q: {r['question']}\nA: {r['answer']}\n\n"

    # 4. 최종 답변 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""아래 면접 Q&A를 참고해서 질문에 답변해주세요.

[참고 Q&A]
{context}

[질문]
{user_query}

답변 시 참고한 Q&A의 출처(질문)를 마지막에 표기해주세요."""
        }]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_queries = [
        "overfitting 방지 방법",
        "TCP UDP 차이",
        "GIL이란",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"질문: {query}")
        print("="*50)
        answer = generate_answer(query)
        print(answer)